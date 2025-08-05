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
This function interpolates polarimetric antenna field patterns for a given set of azimuth and
elevation angles.

## Usage:

```
from quadriga_lib import arrayant

# Minimal example
vr,vi,hr,hi = arrayant.interpolate(arrayant, azimuth, elevation)

# Output as complex type
v,h = arrayant.interpolate(arrayant, azimuth, elevation, complex=1)

# Generate projected distance
vr,vi,hr,hi,dist = arrayant.interpolate(arrayant, azimuth, elevation, dist=1)
v,h,dist = arrayant.interpolate(arrayant, azimuth, elevation, complex=1, dist=1)

# Additional inputs
vr,vi,hr,hi = arrayant.interpolate(arrayant, azimuth, elevation, element, orientation, element_pos)

# Output angles in antenna-local coordinates
vr,vi,hr,hi,az_local,el_local,gamma = arrayant.interpolate(arrayant, azimuth, elevation, orientation=ori, local_angles=1)
```

## Input Arguments:
- **`arrayant`** (required)<br>
  Dictionary containing array antenna data with at least the following keys:
  `e_theta_re`     | Real part of e-theta field component             | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_theta_im`     | Imaginary part of e-theta field component        | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_phi_re`       | Real part of e-phi field component               | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_phi_im`       | Imaginary part of e-phi field component          | Shape: `(n_elevation, n_azimuth, n_elements)`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted        | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions, optional      | Shape: `(3, n_elements)`

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
  If this parameter is not given, element positions `arrayant` are used. If the `arrayant` has no
  positions, they are initialzed to [0,0,0]. For example, when duplicating the fist element by setting
  `element = [1,1)`, different element positions can be set for the  two elements in the output.
  Shape: `(3, n_out)` or empty `()`

- **`complex`** (optional flag)<br>
  If set to 1, output is returned in complex notation. This reduces performance due to additional
  copies of the data in memory. Default: 0, false

- **`dist`** (optional flag)<br>
  Switch to calculate the effective distances for phase calculation. Default: 0, false

- **`local_angles`** (optional flag)<br>
  Switch to return the angles in antenna-local coordinates. These differ from the input when the
  orientation of the antenna is adjusted. Default: 0, false

- **`fast_access`** (optional flag)<br>
  If arrayant data is provided as munpy.ndarray of type double in Fortran-continguous (column-major)
  order, `arrayant_interpolate` can access the Python memory directly without a conversion of the
  data. This will increase performance and is done by default. If the data is not in the correct
  format, a conversion is done in the background. Setting `fast_access` to 1 will skip the conversion
  and throw an error if the arrayant data is not correctly formatted. Default: 0, false (convert)

## Derived inputs:
  `n_azimuth`      | Number of azimuth angles in the filed pattern
  `n_elevation`    | Number of elevation angles in the filed pattern
  `n_elements`     | Number of antenna elements filed pattern of the array antenna
  `n_ang`          | Number of interpolation angles
  `n_out`          | Number of antenna elements in the generated output (may differ from n_elements)

## Output Arguments:
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
  incident path. The distance is calculated by an projection of the array positions on the normal
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
  Polarization rotation angles in [rad], Only returned when `local_angles` flag is set to 1.
  Shape `(n_out, n_ang)`
MD!*/

py::tuple arrayant_interpolate(const py::dict &arrayant,                // Array antenna data
                               const py::array_t<double> &azimuth,      // Azimuth angles in [rad], Shape: `[1, n_ang]` or `[n_out, n_ang]`
                               const py::array_t<double> &elevation,    // Elevation angles in [rad], Shape: `[1, n_ang]` or `[n_out, n_ang]`
                               const py::array_t<arma::uword> &element, // Antenna element indices, 0-based
                               const py::array_t<double> &orientation,  // Euler angles
                               const py::array_t<double> &element_pos,  // Alternative positions of the array antenna elements
                               bool complex,                            // Switch to return outout in complex form or separate Re/Im
                               bool dist,                               // Switch to calculate the effective distances
                               bool local_angles,                       // Switch to calculate the antenna-local angles (az, el, gamma)
                               bool fast_access)                        // Enforces fast memory access
{
    const auto ant = qd_python_dict2arrayant(arrayant, true, fast_access);
    const auto az = qd_python_numpy2arma_Mat(azimuth, true);
    const auto el = qd_python_numpy2arma_Mat(elevation, true);
    const arma::uvec element_ind = qd_python_numpy2arma_Col(element, true);
    const auto ori = qd_python_numpy2arma_Cube(orientation, true);
    const auto elpos = qd_python_numpy2arma_Mat(element_pos, true);

    if (az.n_elem == 0)
        throw std::invalid_argument("Azimuth angles cannot be empty.");

    // Allocate output memory
    arma::uword n_ang = az.n_cols;
    arma::uword n_out = (element_ind.n_elem == 0) ? ant.n_elements() : element_ind.n_elem;

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

    // Interpolate data
    if (dist && local_angles)
        ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos, &dist_proj, &azimuth_loc, &elevation_loc, &gamma);
    else if (dist)
        ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos, &dist_proj);
    else
        ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos);

    // Assemble output
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