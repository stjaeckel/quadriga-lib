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

#include <stdexcept>
#include <cstring>   // std::memcopy
#include <iomanip>   // std::setprecision
#include <algorithm> // std::replace
#include <iostream>

#include "quadriga_arrayant.hpp"
#include "quadriga_tools.hpp"
#include "qd_arrayant_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Array antenna class
SECTION!*/

/*!MD
# arrayant<++>
Class for storing and manipulating array antenna models

## Description:
- An array antenna consists of multiple individual elements.
- Each element occupies a specific position relative to the array's phase-center, its local origin.
- Elements can also be inter-coupled, represented by a coupling matrix.

## Attributes:
`arma::Cube<dtype> e_theta_re`    | Vertical component of the electric field, real part
`arma::Cube<dtype> e_theta_im`    | Vertical component of the electric field, imaginary part
`arma::Cube<dtype> e_phi_re`      | Horizontal component of the electric field, real part
`arma::Cube<dtype> e_phi_im`      | Horizontal component of the electric field, imaginary part
`arma::Col<dtype> azimuth_grid`   | Azimuth angles in pattern (theta) in [rad], between -pi and pi, sorted
`arma::Col<dtype> elevation_grid` | Elevation angles in pattern (phi) in [rad], between -pi/2 and pi/2, sorted
`arma::Mat<dtype> element_pos`    | Element positions (optional), Size: Empty or [3, n_elements]
`arma::Mat<dtype> coupling_re`    | Coupling matrix, real part (optional), Size: [n_elements, n_ports]
`arma::Mat<dtype> coupling_im`    | Coupling matrix, imaginary part (optional), Size: [n_elements, n_ports]
`dtype center_frequency`          | Center frequency in [Hz]

- Allowed datatypes (`dtype`): `float` and `double`
- `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` must have size `[n_elevation, n_azimuth, n_elements]`

## Example:
```
float pi = arma::datum::pi;

quadriga_lib::arrayant<float> ant;
ant.azimuth_grid = {-0.75f * pi, 0.0f, 0.75f * pi, pi};
ant.elevation_grid = {-0.45f * pi, 0.0f, 0.45f * pi};

arma::mat A = arma::linspace(1.0, 12.0, 12);
A.reshape(3, 4);

arma::fcube B;
B.zeros(3, 4, 1);
B.slice(0) = arma::conv_to<arma::fmat>::from(A);

ant.e_theta_re = B * 0.5f;
ant.e_theta_im = B * 0.002f;
ant.e_phi_re = -B;
ant.e_phi_im = -B * 0.001f;

arma::fmat C = {1.0f, 2.0f, 4.0f};
ant.element_pos = C.t();

ant.coupling_re = {1.0f};
ant.coupling_im = {0.1f};
ant.center_frequency = 2.0e9f;
ant.name = "name";
```

## Simple member functions:
`.n_elevation()` | Returns number of elevation angles as 64bit integer
`.n_azimuth()`   | Returns number of azimuth angles as 64bit integer
`.n_elements()`  | Returns number of antenna elements as 64bit integer
`.n_ports()`     | Returns number of ports (after coupling) as 64bit integer
`.copy()`        | Creates a copy of the array antenna object
`.reset()`       | Reset the size to zero (the arrayant object will contain no data)
`.is_valid()`    | Returns an empty string if arrayant object is valid or an error message otherwise

## Complex member fuctions:
[[.append]]                 | Append elements of another antenna array
[[.calc_directivity_dbi]]   | Calculate the directivity (in dBi) of array antenna elements
[[.combine_pattern]]        | Calculate effective radiation patterns for array antennas
[[.copy_element]]           | Creates a copy of a single array antenna element
[[.export_obj_file]]        | Export antenna pattern geometry to Wavefront OBJ file
[[.interpolate]]            | Interpolate array antenna field patterns
[[.qdant_write]]            | Write array antenna object and layout to QDANT file
[[.remove_zeros]]           | Remove zeros from antenna pattern data
[[.rotate_pattern]]         | Adjust orientation of antenna patterns
[[.set_size]]               | Change size of antenna array object
[[.is_valid]]               | Validate integrity of antenna array object
MD!*/

template <typename dtype>
arma::uword quadriga_lib::arrayant<dtype>::n_elevation() const
{
    return e_theta_re.n_rows;
}
template <typename dtype>
arma::uword quadriga_lib::arrayant<dtype>::n_azimuth() const
{
    return e_theta_re.n_cols;
}
template <typename dtype>
arma::uword quadriga_lib::arrayant<dtype>::n_elements() const
{
    return e_theta_re.n_slices;
}
template <typename dtype>
arma::uword quadriga_lib::arrayant<dtype>::n_ports() const
{
    if (coupling_re.empty() && coupling_im.empty())
        return e_theta_re.n_slices;
    else if (coupling_re.empty())
        return coupling_im.n_cols;
    else
        return coupling_re.n_cols;
}

/*!MD
# .append
Append elements of another antenna array

## Description:
- Combines elements of another antenna array (`new_arrayant`) with the current antenna array object.
- Returns a new `arrayant` object containing elements from both antenna arrays.
- Throws an error if the sampling grids of the two antenna arrays do not match.
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::append(const arrayant<dtype> *new_arrayant) const;
```

## Arguments:
- `const arrayant<dtype> *new_arrayant` (input)<br>
  Pointer to an antenna array object whose elements will be added to the current object. Sampling grids must match exactly.

## Returns:
- `quadriga_lib::arrayant<dtype>`<br>
  A new antenna array object combining the current and new antenna elements.

## Example:
```
quadriga_lib::arrayant<double> ant1 = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
quadriga_lib::arrayant<double> ant2 = quadriga_lib::generate_arrayant_custom<double>(120.0, 60.0, 0.0);
quadriga_lib::arrayant<double> combined_ant = ant1.append(&ant2);
```
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::append(const quadriga_lib::arrayant<dtype> *new_arrayant) const
{
    // Check if arrayant objects are valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());
    error_message = new_arrayant->is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Check if grids are identical
    bool eq = arma::approx_equal(azimuth_grid, new_arrayant->azimuth_grid, "absdiff", (dtype)1.0e-6);
    if (!eq)
    {
        error_message = "Azimuth sampling grids don't match.";
        throw std::invalid_argument(error_message.c_str());
    }
    eq = arma::approx_equal(elevation_grid, new_arrayant->elevation_grid, "absdiff", (dtype)1.0e-6);
    if (!eq)
    {
        error_message = "Elevation sampling grids don't match.";
        throw std::invalid_argument(error_message.c_str());
    }

    arma::uword n_elevation = this->n_elevation();
    arma::uword n_azimuth = this->n_azimuth();
    arma::uword n_elements_1 = this->n_elements();
    arma::uword n_elements_2 = new_arrayant->n_elements();
    arma::uword n_ports_1 = this->n_ports();
    arma::uword n_ports_2 = new_arrayant->n_ports();

    quadriga_lib::arrayant<dtype> output;
    output.set_size(n_elevation, n_azimuth, n_elements_1 + n_elements_2, n_ports_1 + n_ports_2);

    // Copy data from first antenna
    std::memcpy(output.azimuth_grid.memptr(), this->azimuth_grid.memptr(), n_azimuth * sizeof(dtype));
    std::memcpy(output.elevation_grid.memptr(), this->elevation_grid.memptr(), n_elevation * sizeof(dtype));
    std::memcpy(output.e_theta_re.slice_memptr(0), this->e_theta_re.slice_memptr(0), n_azimuth * n_elevation * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_theta_im.slice_memptr(0), this->e_theta_im.slice_memptr(0), n_azimuth * n_elevation * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_phi_re.slice_memptr(0), this->e_phi_re.slice_memptr(0), n_azimuth * n_elevation * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_phi_im.slice_memptr(0), this->e_phi_im.slice_memptr(0), n_azimuth * n_elevation * n_elements_1 * sizeof(dtype));
    std::memcpy(output.element_pos.colptr(0), this->element_pos.colptr(0), 3 * n_elements_1 * sizeof(dtype));
    for (arma::uword n = 0; n < n_ports_1; ++n)
    {
        std::memcpy(output.coupling_re.colptr(n), this->coupling_re.colptr(n), n_elements_1 * sizeof(dtype));
        std::memcpy(output.coupling_im.colptr(n), this->coupling_im.colptr(n), n_elements_1 * sizeof(dtype));
    }

    // Copy data from second antenna
    std::memcpy(output.e_theta_re.slice_memptr(n_elements_1), new_arrayant->e_theta_re.slice_memptr(0), n_azimuth * n_elevation * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_theta_im.slice_memptr(n_elements_1), new_arrayant->e_theta_im.slice_memptr(0), n_azimuth * n_elevation * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_phi_re.slice_memptr(n_elements_1), new_arrayant->e_phi_re.slice_memptr(0), n_azimuth * n_elevation * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_phi_im.slice_memptr(n_elements_1), new_arrayant->e_phi_im.slice_memptr(0), n_azimuth * n_elevation * n_elements_2 * sizeof(dtype));
    std::memcpy(output.element_pos.colptr(n_elements_1), new_arrayant->element_pos.colptr(0), 3 * n_elements_2 * sizeof(dtype));
    for (arma::uword n = 0; n < n_ports_2; ++n)
    {
        std::memcpy(output.coupling_re.colptr(n + n_ports_1) + n_elements_1, new_arrayant->coupling_re.colptr(n), n_elements_2 * sizeof(dtype));
        std::memcpy(output.coupling_im.colptr(n + n_ports_1) + n_elements_1, new_arrayant->coupling_im.colptr(n), n_elements_2 * sizeof(dtype));
    }

    return output;
}

/*!MD
# .calc_directivity_dBi
Calculate the directivity (in dBi) of array antenna elements

## Description:
- Member function of [[arrayant]]
- Directivity is a parameter of an antenna or which measures the degree to which the radiation emitted
  is concentrated in a single direction. It is the ratio of the radiation intensity in a given direction
  from the antenna to the radiation intensity averaged over all directions. Therefore, the directivity
  of a hypothetical isotropic radiator is 1, or 0 dBi.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
dtype quadriga_lib::arrayant<dtype>::calc_directivity_dBi(arma::uword i_element) const;
```
## Arguments:
- `arma::uword **i_element**`<br>
  Element index, 0-based<br>

## Example:
```
auto ant = quadriga_lib::generate_arrayant_dipole<float>();
float directivity = ant.calc_directivity_dBi( 0 );
```
MD!*/

template <typename dtype>
dtype quadriga_lib::arrayant<dtype>::calc_directivity_dBi(arma::uword i_element) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() == 0 && i_element >= n_elements())
        error_message = "Element index out of bound.";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Constants
    double pi2 = 2.0 * arma::datum::pi, pi_half = 0.5 * arma::datum::pi;
    arma::uword naz = azimuth_grid.n_elem, nel = elevation_grid.n_elem;

    // Az and El grid as double
    arma::vec az = arma::conv_to<arma::vec>::from(azimuth_grid);
    arma::vec el = arma::conv_to<arma::vec>::from(elevation_grid);

    // Calculate the azimuth weights
    arma::vec waz(naz, arma::fill::none);
    double *ptr = waz.memptr();
    for (arma::uword i = 0; i < naz; ++i)
    {
        double x = i == 0 ? az.at(naz - 1) - pi2 : az.at(i - 1);
        double y = i == naz - 1 ? az.at(0) + pi2 : az.at(i + 1);
        ptr[i] = y - x;
    }

    // Calculate the elevation weights
    arma::vec wel(nel, arma::fill::none);
    ptr = wel.memptr();
    for (arma::uword i = 0; i < nel; ++i)
    {
        double x = i == 0 ? -pi_half : 0.5 * el.at(i - 1) + 0.5 * el.at(i);
        double y = i == nel - 1 ? pi_half : 0.5 * el.at(i) + 0.5 * elevation_grid.at(i + 1);
        arma::vec tmp = arma::linspace<arma::vec>(x, y, 21);
        double val = arma::accu(arma::cos(tmp));
        ptr[i] = val * (y - x);
    }

    // Combine weights
    arma::mat W(nel, naz, arma::fill::none);
    ptr = W.memptr();
    double norm = 0.0;
    for (double *col = waz.begin(); col != waz.end(); ++col)
        for (double *row = wel.begin(); row != wel.end(); ++row)
            *ptr = *row * *col, norm += *ptr++;
    ptr = W.memptr();
    norm = 1.0 / norm;
    for (arma::uword i = 0; i < naz * nel; ++i)
        ptr[i] *= norm;

    // Calculate the directivity
    double p_sum = 0.0, p_max = 0.0;
    const dtype *p_theta_re = e_theta_re.slice_memptr(i_element), *p_theta_im = e_theta_im.slice_memptr(i_element);
    const dtype *p_phi_re = e_phi_re.slice_memptr(i_element), *p_phi_im = e_phi_im.slice_memptr(i_element);
    for (arma::uword i = 0; i < naz * nel; ++i)
    {
        double a = double(p_theta_re[i]), b = double(p_theta_im[i]), c = double(p_phi_re[i]), d = double(p_phi_im[i]);
        double pow = a * a + b * b + c * c + d * d;
        p_max = pow > p_max ? pow : p_max;
        p_sum += pow * ptr[i];
    }

    double directivity = p_max < 1.0e-14 ? 0.0 : 10.0 * std::log10(p_max / p_sum);
    return dtype(directivity);
}

/*!MD
# .combine_pattern
Calculate effective radiation patterns for array antennas

## Description:
- Member function of [[arrayant]]
- By integrating element radiation patterns, element positions, and the coupling weights, one can
  determine an effective radiation pattern observable by a receiver in the antenna's far field.
- Leveraging these effective patterns is especially beneficial in antenna design, beamforming
  applications such as in 5G systems, and in planning wireless communication networks in complex
  environments like urban areas. This streamlined approach offers a significant boost in computation
  speed when calculating MIMO channel coefficients, as it reduces the number of necessary operations.
- Allowed datatypes (`dtype`): `float` and `double`

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::combine_pattern(
                const arma::Col<dtype> *azimuth_grid_new = nullptr,
                const arma::Col<dtype> *elevation_grid_new = nullptr) const;
```

## Arguments:
- `arma::Col<dtype> ***azimuth_grid_new**` (optional)<br>
  Azimuth angle grid of the output array antenna in [rad], between -pi and pi, sorted

- `arma::Col<dtype> ***elevation_grid_new**` (optional)<br>
  Elevation angle grid of the output array antenna in [rad], between -pi/2 and pi/2, sorted

## Example:
```
auto ant = quadriga_lib::generate_arrayant_omni<double>();  // Generate omni antenna
ant.copy_element(0, 1);                                     // Duplicate the first element
ant.element_pos.row(1) = {-0.25, 0.25};                     // Set element positions (in lambda)
ant.coupling_re.ones(2, 1);                                 // Set coupling matrix (real part)
ant.coupling_im.reset();                                    // Remove imaginary part
ant = ant.combine_pattern();                                // Calculate the combined pattern
```
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::combine_pattern(const arma::Col<dtype> *azimuth_grid_new,
                                                                             const arma::Col<dtype> *elevation_grid_new) const
{
    // Check if calling arrayant object is valid
    std::string error_message = this->is_valid(false); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Get output angular grid
    bool new_azgrid = azimuth_grid_new != nullptr && azimuth_grid_new->n_elem != 0;
    bool new_elgrid = elevation_grid_new != nullptr && elevation_grid_new->n_elem != 0;

    if (new_azgrid && !qd_in_range(azimuth_grid_new->memptr(), azimuth_grid_new->n_elem, dtype(-3.1415930), dtype(3.1415930), true, true))
        throw std::invalid_argument("Values of 'azimuth_grid_new' must be sorted and in between -pi and pi (equivalent to -180 to 180 degree).");

    if (new_elgrid && !qd_in_range(elevation_grid_new->memptr(), elevation_grid_new->n_elem, dtype(-1.5707965), dtype(1.5707965), true, true))
        throw std::invalid_argument("Values of 'elevation_grid_new' must be sorted and in between -pi/2 and pi/2 (equivalent to -90 to 90 degree).");

    const dtype *p_azimuth_grid = (new_azgrid) ? azimuth_grid_new->memptr() : this->azimuth_grid.memptr();
    const dtype *p_elevation_grid = (new_elgrid) ? elevation_grid_new->memptr() : this->elevation_grid.memptr();

    arma::uword n_azimuth_out = (new_azgrid) ? azimuth_grid_new->n_elem : this->azimuth_grid.n_elem;
    arma::uword n_elevation_out = (new_elgrid) ? elevation_grid_new->n_elem : this->elevation_grid.n_elem;
    arma::uword n_ang = n_azimuth_out * n_elevation_out;

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    {
        dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr();
        for (arma::uword ia = 0ULL; ia < n_azimuth_out; ++ia)
            for (arma::uword ie = 0ULL; ie < n_elevation_out; ++ie)
                *p_azimuth++ = p_azimuth_grid[ia], *p_elevation++ = p_elevation_grid[ie];
    }

    // Get element positions
    arma::uword n_elements = this->e_theta_re.n_slices;
    arma::Mat<dtype> element_pos_empty(3, n_elements, arma::fill::zeros);
    const auto p_element_pos = this->element_pos.empty() ? &element_pos_empty : &this->element_pos;

    // Interpolate the pattern data
    unsigned n32_out = unsigned(n_elements);
    arma::Col<unsigned> i_element = arma::linspace<arma::Col<unsigned>>(1, n32_out, n32_out);
    arma::Cube<dtype> orientation(3, 1, 1);
    arma::Mat<dtype> V_re(n_elements, n_ang), V_im(n_elements, n_ang), H_re(n_elements, n_ang), H_im(n_elements, n_ang), dist(n_elements, n_ang);
    arma::Mat<dtype> EMPTY;

    qd_arrayant_interpolate(&this->e_theta_re, &this->e_theta_im, &this->e_phi_re, &this->e_phi_im,
                            &this->azimuth_grid, &this->elevation_grid, &azimuth, &elevation,
                            &i_element, &orientation, p_element_pos,
                            &V_re, &V_im, &H_re, &H_im, &dist, &EMPTY, &EMPTY, &EMPTY);

    // Apply phase shift caused by element positions
    double lambda = 299792448.0 / (double)this->center_frequency;
    dtype wave_no = dtype(2.0 * arma::datum::pi / lambda);
    arma::Mat<std::complex<dtype>> phase(arma::cos(wave_no * dist), arma::sin(-wave_no * dist));
    arma::Mat<std::complex<dtype>> Vi(V_re, V_im), Hi(H_re, H_im);
    Vi = Vi % phase, Hi = Hi % phase;

    // Apply coupling
    arma::uword n_ports_out = this->coupling_re.empty() ? n_elements : this->coupling_re.n_cols;
    arma::Mat<std::complex<dtype>> Vo(n_ang, n_ports_out), Ho(n_ang, n_ports_out);
    if (!this->coupling_re.empty())
    {
        arma::Mat<dtype> coupling_im_empty(n_elements, n_ports_out, arma::fill::zeros);
        const auto p_coupling_im = this->coupling_im.empty() ? &coupling_im_empty : &this->coupling_im;
        arma::Mat<std::complex<dtype>> coupling(this->coupling_re, *p_coupling_im);

        for (arma::uword i = 0ULL; i < n_elements; ++i)
        {
            arma::Col<std::complex<dtype>> vi = Vi.row(i).as_col(), hi = Hi.row(i).as_col();
            for (arma::uword o = 0ULL; o < n_ports_out; ++o)
            {
                std::complex<dtype> cpl = coupling.at(i, o);
                Vo.col(o) += vi * cpl, Ho.col(o) += hi * cpl;
            }
        }
    }
    else
        Vo = Vi, Ho = Hi;

    // Write output data
    quadriga_lib::arrayant<dtype> output;
    {
        output.set_size(n_elevation_out, n_azimuth_out, n_ports_out, n_ports_out);

        std::memcpy(output.azimuth_grid.memptr(), p_azimuth_grid, n_azimuth_out * sizeof(dtype));
        std::memcpy(output.elevation_grid.memptr(), p_elevation_grid, n_elevation_out * sizeof(dtype));

        arma::Mat<dtype> cpy = arma::real(Vo);
        arma::uword n_bytes = n_elevation_out * n_azimuth_out * n_ports_out * sizeof(dtype);
        std::memcpy(output.e_theta_re.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::imag(Vo);
        std::memcpy(output.e_theta_im.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::real(Ho);
        std::memcpy(output.e_phi_re.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::imag(Ho);
        std::memcpy(output.e_phi_im.memptr(), cpy.memptr(), n_bytes);

        output.element_pos.zeros();
        output.coupling_re.eye();
        output.coupling_im.zeros();
        output.center_frequency = this->center_frequency;
        output.name = name;

        // Set the data pointers for the quick check.
        output.check_ptr[0] = output.e_theta_re.memptr();
        output.check_ptr[1] = output.e_theta_im.memptr();
        output.check_ptr[2] = output.e_phi_re.memptr();
        output.check_ptr[3] = output.e_phi_im.memptr();
        output.check_ptr[4] = output.azimuth_grid.memptr();
        output.check_ptr[5] = output.elevation_grid.memptr();
        output.check_ptr[6] = output.element_pos.memptr();
        output.check_ptr[7] = output.coupling_re.memptr();
        output.check_ptr[8] = output.coupling_im.memptr();
    }

    return output;
}

// Creates a copy of the array antenna object
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::copy() const
{
    quadriga_lib::arrayant<dtype> ant;

    ant.name = name;
    ant.e_theta_re = e_theta_re;
    ant.e_theta_im = e_theta_im;
    ant.e_phi_re = e_phi_re;
    ant.e_phi_im = e_phi_im;
    ant.azimuth_grid = azimuth_grid;
    ant.elevation_grid = elevation_grid;
    ant.element_pos = element_pos;
    ant.coupling_re = coupling_re;
    ant.coupling_im = coupling_im;
    ant.center_frequency = center_frequency;
    ant.read_only = false;

    return ant;
}

/*!MD
# .copy_element
Creates a copy of a single array antenna element

## Description:
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` and `double`

## Declaration:
```
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uvec destination);

void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uword destination);
```

## Arguments:
- `arma::uword **source**` (optional)<br>
  Index of the source element (0-based)

- `arma::uvec **destination**` or `arma::uword **destination**`<br>
  Index of the destinations element (0-based), either as a vector or as a scalar.

## Example:
```
auto ant = quadriga_lib::generate_arrayant_omni<double>();  // Generate omni antenna
ant.copy_element(0, 1);                                     // Duplicate the first element
ant.copy_element(1, {2,3});                                 // Duplicate multiple times
```
MD!*/

// Copy antenna elements, enlarge array size if needed
template <typename dtype>
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uvec destination)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    arma::uword n_el = e_theta_re.n_rows;
    arma::uword n_az = e_theta_re.n_cols;
    arma::uword n_ang = n_el * n_az;
    arma::uword n_elements = e_theta_re.n_slices;
    arma::uword n_element_max = destination.max() + 1;
    arma::uword n_ports = coupling_re.n_cols;

    if (source >= n_elements)
        error_message = "Source index out of bound";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Enlarge existing arrayant object
    if (n_element_max > n_elements)
    {
        arma::uword added_elements = n_element_max - n_elements;
        e_theta_re.resize(n_el, n_az, n_element_max);
        e_theta_im.resize(n_el, n_az, n_element_max);
        e_phi_re.resize(n_el, n_az, n_element_max);
        e_phi_im.resize(n_el, n_az, n_element_max);
        element_pos.resize(3, n_element_max);
        coupling_re.resize(n_element_max, n_ports + added_elements);
        coupling_im.resize(n_element_max, n_ports + added_elements);
        for (arma::uword i = 0; i < added_elements; ++i)
            coupling_re.at(n_elements + i, n_ports + i) = (dtype)1.0;
    }

    // Copy data from source to destination
    for (auto dest = destination.begin(); dest != destination.end(); ++dest)
        if (source != *dest)
        {
            std::memcpy(e_theta_re.slice_memptr(*dest), e_theta_re.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_theta_im.slice_memptr(*dest), e_theta_im.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_phi_re.slice_memptr(*dest), e_phi_re.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_phi_im.slice_memptr(*dest), e_phi_im.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(element_pos.colptr(*dest), element_pos.colptr(source), 3 * sizeof(dtype));
        }

    // Set the data pointers for the quick check.
    check_ptr[0] = e_theta_re.memptr();
    check_ptr[1] = e_theta_im.memptr();
    check_ptr[2] = e_phi_re.memptr();
    check_ptr[3] = e_phi_im.memptr();
    check_ptr[4] = azimuth_grid.memptr();
    check_ptr[5] = elevation_grid.memptr();
    check_ptr[6] = element_pos.memptr();
    check_ptr[7] = coupling_re.memptr();
    check_ptr[8] = coupling_im.memptr();
}

template <typename dtype>
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uword destination)
{
    arma::uvec dest(1);
    dest.at(0) = destination;
    copy_element(source, dest);
}

/*!MD
# .export_obj_file
Export antenna pattern geometry to Wavefront OBJ file

## Description:
- This function exports the antenna pattern geometry to a Wavefront OBJ file, useful for visualization in 3D software such as Blender.
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant<dtype>::export_obj_file(
                std::string fn,
                dtype directivity_range = 30.0,
                std::string colormap = "jet",
                dtype object_radius = 1.0,
                arma::uword icosphere_n_div = 4,
                arma::uvec i_element = {}) const;
```

## Arguments:
- `std::string **fn**` (input)<br>
  Filename of the OBJ file to which the antenna pattern will be exported. Cannot be empty.

- `dtype **directivity_range** = 30.0` (optional input)<br>
  Directivity range in decibels (dB) for visualizing the antenna pattern. This value defines the
  dynamic range of the visualized directivity pattern. Default: `30.0`

- `std::string **colormap** = "jet"` (optional input)<br>
  Colormap used for visualizing the antenna directivity. Supported colormaps are: `jet`,
  `parula`, `winter`, `hot`, `turbo`, `copper`, `spring`, `cool`, `gray`, `autumn`, `summer`.
  Default: `"jet"`

- `dtype **object_radius** = 1.0` (optional input)<br>
  Radius of the exported antenna pattern geometry object, specified in meters. Default: `1.0`

- `arma::uword **icosphere_n_div** = 4` (optional input)<br>
  Number of subdivisions used to map the antenna pattern onto an icosphere. Higher values yield finer
  mesh resolution. Default: `4`

- `arma::uvec **i_element** = {}` (optional input)<br>
  Antenna element indices for which the pattern geometry is exported. Indices are 0-based. Providing
  an empty vector `{}` (default) exports the geometry for all elements of the antenna array.

## Example:
```
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
ant.export_obj_file("antenna_pattern.obj", 40.0, "turbo", 1.5, 5);
```
MD!*/

// ARRAYANT METHOD : OBJ Export
template <typename dtype>
void quadriga_lib::arrayant<dtype>::export_obj_file(std::string fn, dtype directivity_range, std::string colormap,
                                                    dtype object_radius, arma::uword icosphere_n_div,
                                                    arma::uvec i_element) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Input validation
    std::string fn_suffix = ".obj";
    std::string fn_mtl;

    if (fn.size() >= fn_suffix.size() &&
        fn.compare(fn.size() - fn_suffix.size(), fn_suffix.size(), fn_suffix) == 0)
    {
        fn_mtl = fn.substr(0, fn.size() - fn_suffix.size()) + ".mtl";
    }
    else
        throw std::invalid_argument("OBJ-File name must end with .obj");

    // Replace spaces in ant_name
    std::string ant_name = name;
    std::replace(ant_name.begin(), ant_name.end(), ' ', '_');

    // Extract the file name from the path
    std::string fn_mtl_base;
    size_t pos = fn_mtl.find_last_of("/");
    if (pos != std::string::npos)
        fn_mtl_base = fn_mtl.substr(pos + 1);
    else
        fn_mtl_base = fn_mtl;

    double range = -std::abs((double)directivity_range);
    if (range == 0.0)
        throw std::invalid_argument("Directivity range cannot be 0.");

    double radius = (double)object_radius;
    if (radius <= 0.0)
        throw std::invalid_argument("Object radius must be larger than 0.");

    arma::uword no_elements = n_elements();

    if (i_element.n_elem == 0)
        i_element = arma::regspace<arma::uvec>(0, no_elements - 1);

    if (arma::any(i_element >= no_elements))
        throw std::invalid_argument("Element indices 'i_element' cannot exceed the array antenna size.");

    size_t no_elements_t = (size_t)i_element.n_elem;

    // Colormap
    arma::uchar_mat cmap = quadriga_lib::colormap(colormap);
    size_t n_cmap = (size_t)cmap.n_rows;

    // Export colormap to material file
    std::ofstream outFile(fn_mtl);
    if (outFile.is_open())
    {
        // Write some text to the file
        outFile << "# QuaDRiGa " << "antenna pattern colormap\n\n";
        for (size_t i = 0; i < n_cmap; ++i)
        {
            double R = (double)cmap(i, 0) / 255.0;
            double G = (double)cmap(i, 1) / 255.0;
            double B = (double)cmap(i, 2) / 255.0;
            outFile << "newmtl QuaDRiGa_ANT_" << colormap << "_" << std::setfill('0') << std::setw(2) << i << "\n";
            outFile << std::fixed << std::setprecision(6) << "Kd " << R << " " << G << " " << B << "\n\n";
        }
        outFile.close();
    }

    // Generate vertices and faces
    arma::mat vertices_icosphere; // double precision
    arma::u32_mat faces_icosphere;
    size_t no_faces, no_vert;
    {
        arma::mat center, vert;
        no_faces = quadriga_lib::icosphere<double>(icosphere_n_div, 1.0, &center, nullptr, &vert);

        vertices_icosphere.set_size(3 * no_faces, 3);
        faces_icosphere.set_size(no_faces, 3);

        double *pc = center.memptr(), *pv = vert.memptr(), *v = vertices_icosphere.memptr();
        unsigned *f_ico = faces_icosphere.memptr();

        for (size_t i_face = 0; i_face < no_faces; ++i_face)
        {
            double x, y, z;

            x = pc[i_face] + pv[i_face];
            y = pc[i_face + no_faces] + pv[i_face + no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 2 * no_faces];

            v[3 * i_face] = x;
            v[3 * i_face + 3 * no_faces] = y;
            v[3 * i_face + 6 * no_faces] = z;
            f_ico[i_face] = 3 * (unsigned)i_face + 1;

            x = pc[i_face] + pv[i_face + 3 * no_faces];
            y = pc[i_face + no_faces] + pv[i_face + 4 * no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 5 * no_faces];

            v[3 * i_face + 1] = x;
            v[3 * i_face + 3 * no_faces + 1] = y;
            v[3 * i_face + 6 * no_faces + 1] = z;
            f_ico[i_face + no_faces] = 3 * (unsigned)i_face + 2;

            x = pc[i_face] + pv[i_face + 6 * no_faces];
            y = pc[i_face + no_faces] + pv[i_face + 7 * no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 8 * no_faces];

            v[3 * i_face + 2] = x;
            v[3 * i_face + 3 * no_faces + 2] = y;
            v[3 * i_face + 6 * no_faces + 2] = z;
            f_ico[i_face + 2 * no_faces] = 3 * (unsigned)i_face + 3;
        }
    }
    no_vert = 3 * no_faces;

    // Convert vertices to angles
    arma::Mat<dtype> azimuth, elevation;
    azimuth.set_size(1, no_vert);
    elevation.set_size(1, no_vert);
    {
        double *v = vertices_icosphere.memptr();
        dtype *az = azimuth.memptr(), *el = elevation.memptr();
        for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
        {
            double x = v[i_vert];
            double y = v[i_vert + no_vert];
            double z = v[i_vert + 2 * no_vert];
            x = x > 1.0 ? 1.0 : x, y = y > 1.0 ? 1.0 : y, z = z > 1.0 ? 1.0 : z;
            x = x < -1.0 ? -1.0 : x, y = y < -1.0 ? -1.0 : y, z = z < -1.0 ? -1.0 : z;
            az[i_vert] = (dtype)std::atan2(y, x);
            el[i_vert] = (dtype)std::asin(z);
        }
    }

    // Interpolate antenna patterns (for all elements in the array)
    arma::Mat<dtype> V_re, V_im, H_re, H_im;
    interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, i_element);
    dtype *vr = V_re.memptr(), *vi = V_im.memptr(), *hr = H_re.memptr(), *hi = H_im.memptr();

    // Calculate maximum Power in the Pattern
    double max_pow = 0.0;
    for (size_t i_el = 0; i_el < no_elements_t; ++i_el)
        for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
        {
            size_t i = i_vert * no_elements_t + i_el;
            double a = (double)vr[i], b = (double)vi[i], c = (double)hr[i], d = (double)hi[i];
            double p = a * a + b * b + c * c + d * d;
            max_pow = (p > max_pow) ? p : max_pow;
        }
    max_pow = 1.0 / max_pow;

    // Write OBJ File
    outFile = std::ofstream(fn);
    if (outFile.is_open())
    {
        outFile << "# QuaDRiGa " << "antenna pattern\n";
        outFile << "mtllib " << fn_mtl_base << "\n";

        // Face offset for previous element in the OBJ file
        unsigned face_offset = 1; // OBJ starts counting at 1

        // Export each element to the OBJ file
        double scale = 1.0 / (-range);
        for (size_t i_el = 0; i_el < no_elements_t; ++i_el)
        {
            arma::uword i_array_element = i_element(i_el);
            outFile << "\no " << "QuaDRiGa_ANT_" << ant_name << "_" << std::setfill('0') << std::setw(3) << i_array_element << "\n";

            // Calculate Shape
            arma::vec pow(no_vert);
            double *pp = pow.memptr();
            for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
            {
                size_t i = i_vert * no_elements_t + i_el;
                double a = (double)vr[i], b = (double)vi[i], c = (double)hr[i], d = (double)hi[i];
                double p = a * a + b * b + c * c + d * d;
                p *= max_pow;                // Scale by maximum power
                p = 10.0 * std::log10(p);    // Convert to dB
                p = (p < range) ? range : p; // Set minimum
                p = (p - range) * scale;     // Set to range 0 ... 1
                pp[i_vert] = p;
            }

            // Scale vertices
            arma::mat scaled_vertices(no_vert, 3);
            double *v_scale = scaled_vertices.memptr();  // Scaled vertices
            double *v_ico = vertices_icosphere.memptr(); // Icosphere vertices
            for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
            {
                v_scale[i_vert] = v_ico[i_vert] * pp[i_vert];                             // x
                v_scale[i_vert + no_vert] = v_ico[i_vert + no_vert] * pp[i_vert];         // y
                v_scale[i_vert + 2 * no_vert] = v_ico[i_vert + 2 * no_vert] * pp[i_vert]; // z
            }

            // Process faces
            double eps = 0.001; // Co-location tolerance in [m], relative to 1 m radius
            eps *= eps;         // Squared value

            arma::mat final_vertices(no_vert, 3);
            arma::u32_mat vert_index(no_faces, 3);
            arma::u32_vec cmap_index(no_faces);

            double *v_final = final_vertices.memptr(); // Final vertices
            unsigned *f_final = vert_index.memptr();   // Final face indices
            unsigned *i_cmap = cmap_index.memptr();    // Colormap index

            size_t no_final_vert = 0; // Number of final vertices
            size_t no_final_face = 0; // Number of final faces

            for (size_t i_face = 0; i_face < no_faces; ++i_face)
            {
                // Load the 3 scaled vertices of the face
                double x0 = v_scale[3 * i_face];
                double y0 = v_scale[3 * i_face + no_vert];
                double z0 = v_scale[3 * i_face + 2 * no_vert];

                double x1 = v_scale[3 * i_face + 1];
                double y1 = v_scale[3 * i_face + no_vert + 1];
                double z1 = v_scale[3 * i_face + 2 * no_vert + 1];

                double x2 = v_scale[3 * i_face + 2];
                double y2 = v_scale[3 * i_face + no_vert + 2];
                double z2 = v_scale[3 * i_face + 2 * no_vert + 2];

                // Check if face vertices collapse to a point or line
                double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                double dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                dx = x2 - x0, dy = y2 - y0, dz = z2 - z0;
                dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
                dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                // Do for each of the 3 vertices
                for (size_t i3 = 0; i3 < 3; ++i3)
                {
                    double x = x0, y = y0, z = z0;
                    if (i3 == 1)
                        x = x1, y = y1, z = z1;
                    else if (i3 == 2)
                        x = x2, y = y2, z = z2;

                    // Add the vertices to the final list
                    size_t i_vert_final = 0; // Face index

                    // Get vertex ID
                    for (size_t i = 0; i < no_final_vert; ++i) // Search exisiting vertex
                    {
                        dx = v_final[i] - x;
                        dy = v_final[i + no_vert] - y;
                        dz = v_final[i + 2 * no_vert] - z;
                        dd = dx * dx + dy * dy + dz * dz;
                        if (dd < eps)
                        {
                            i_vert_final = i;
                            break;
                        }
                    }
                    if (i_vert_final == 0) // Add new vertex to ist
                    {
                        i_vert_final = no_final_vert;
                        v_final[i_vert_final] = x;
                        v_final[i_vert_final + no_vert] = y;
                        v_final[i_vert_final + 2 * no_vert] = z;
                        ++no_final_vert;
                    }

                    // Write face ID
                    f_final[no_final_face + i3 * no_faces] = (unsigned)i_vert_final;
                }

                // Get the colormap index
                double pf = pp[3 * i_face] + pp[3 * i_face + 1] + pp[3 * i_face + 2];
                pf *= 21.0; // 63 * 0.33
                pf = std::round(pf);
                unsigned c = (unsigned)pf;
                c = (c > 63) ? 63 : c;
                i_cmap[no_final_face] = c;

                ++no_final_face;
            }

            // Write vertices to file
            for (size_t i_vert = 0; i_vert < no_final_vert; ++i_vert)
            {
                double x = v_final[i_vert];
                double y = v_final[i_vert + no_vert];
                double z = v_final[i_vert + 2 * no_vert];

                x *= radius;
                y *= radius;
                z *= radius;

                if (element_pos.n_elem != 0)
                {
                    x += element_pos(0, i_array_element);
                    y += element_pos(1, i_array_element);
                    z += element_pos(2, i_array_element);
                }

                outFile << "v " << x << " " << y << " " << z << "\n";
            }

            // Write faces to file, ordered by color
            for (unsigned i_color = 0; i_color < 64; ++i_color)
            {
                bool color_not_used = true;
                for (size_t i_face = 0; i_face < no_final_face; ++i_face)
                {
                    if (i_cmap[i_face] == i_color)
                    {
                        if (color_not_used)
                        {
                            outFile << "usemtl QuaDRiGa_ANT_" << colormap << "_" << std::setfill('0') << std::setw(2) << i_color << "\n";
                            color_not_used = false;
                        }
                        outFile << "f " << f_final[i_face] + face_offset << " " << f_final[i_face + no_faces] + face_offset << " " << f_final[i_face + 2 * no_faces] + face_offset << "\n";
                    }
                }
            }
            face_offset += (unsigned)no_final_vert;
        }

        outFile.close();
    }
}

/*!MD
# .interpolate
Interpolate array antenna field patterns

## Description:
- This function interpolates polarimetric antenna field patterns for a given set of azimuth and
  elevation angles.
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant<dtype>::interpolate(
                const arma::Mat<dtype> *azimuth,
                const arma::Mat<dtype> *elevation,
                arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
                arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
                arma::uvec i_element,
                const arma::Cube<dtype> *orientation,
                const arma::Mat<dtype> *element_pos_i,
                arma::Mat<dtype> *dist,
                arma::Mat<dtype> *azimuth_loc, arma::Mat<dtype> *elevation_loc,
                arma::Mat<dtype> *gamma) const;
```

## Arguments:
- `const arma::Mat<dtype> ***azimuth**` (input)<br>
  Azimuth angles in [rad] for which the field pattern should be interpolated. Values must be
  between -pi and pi, cannot be NULL
  Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
             | Size: `[1, n_ang]`
  Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
             | Size: `[n_out, n_ang]`

- `const arma::Mat<dtype> ***elevation**` (input)<br>
  Elevation angles in [rad] for which the field pattern should be interpolated. Values must be
  between -pi/2 and pi/2, cannot be NULL
  Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
             | Size: `[1, n_ang]`
  Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
             | Size: `[n_out, n_ang]`

- `arma::Mat<dtype> ***V_re**` (output)<br>
  Real part of the interpolated e-theta (vertical) field component, Size `[n_out, n_ang]`,
  will be resized if it does not match the required size (invalidates data pointers), cannot be NULL

- `arma::Mat<dtype> ***V_im**` (output)<br>
  Imaginary part of the interpolated e-theta (vertical) field component, Size `[n_out, n_ang]`
  will be resized if it does not match the required size (invalidates data pointers), cannot be NULL

- `arma::Mat<dtype> ***H_re**` (output)<br>
  Real part of the interpolated e-phi (horizontal) field component, Size `[n_out, n_ang]`
  will be resized if it does not match the required size (invalidates data pointers), cannot be NULL

- `arma::Mat<dtype> ***H_im**` (output)<br>
  Imaginary part of the interpolated e-phi (horizontal) field component, Size `[n_out, n_ang]`
  will be resized if it does not match the required size (invalidates data pointers), cannot be NULL

- `arma::uvec **i_element** = {}` (optional input)<br>
  The element indices for which the interpolation should be done, optional argument,
  values must be between 1 and `n_elements`. It is possible to duplicate elements, i.e. by passing
  `{1,1,2}`. If this parameter is not provided (or an empty array is passed), `i_element` is initialized
  to include all elements of the array antenna. In this case, `n_out = n_elements`,
  Length: `n_out` or  empty `{}`

- `const arma::Cube<dtype> ***orientation** = nullptr` (optional input)<br>
  This (optional) 3-element vector allows for setting orientation of the array antenna or
  of individual elements using Euler angles (bank, tilt, heading); values must be given in [rad];
  By default, the orientation is `{0,0,0}`, i.e. the broadside of the antenna points at the horizon
  towards the East. Sizes: `nullptr` (use default), `[3, 1]` (set orientation for entire array),
  `[3, n_out]` (set orientation for individual elements), or `[3, 1, n_ang]` (set orientation for
  individual angles) or `[3, n_out, n_ang]` (set orientation for individual elements and angles)

- `const arma::Mat<dtype> ***element_pos_i** = nullptr` (optional input)<br>
  Positions of the array antenna elements in local cartesian coordinates (using units
  of [m]). If this parameter is not given, the element positions from the `arrayant` object are used.
  Sizes: `nullptr` (use `arrayant.element_pos`), `[3, n_out]` (set alternative positions)

- `arma::Mat<dtype> ***dist** = nullptr` (optional output)<br>
  The effective distances between the antenna elements when seen from the direction
  of the incident path. The distance is calculated by an projection of the array positions on the normal
  plane of the incident path. This is needed for calculating the phase of the antenna response.
  Size: `nullptr` (do not calculate this) or `[n_out, n_ang]` (argument be resized if it does not already
  match this size)

- `arma::Mat<dtype> ***azimuth_loc** = nullptr` (optional output)<br>
  The azimuth angles in [rad] for the local antenna coordinate system, i.e., after
  applying the `orientation`. If no orientation vector is given, these angles are identical to the input
  azimuth angles. Size: `nullptr` or `[n_out, n_ang]`

- `arma::Mat<dtype> ***elevation_loc** = nullptr` (optional output)<br>
  The elevation angles in [rad] for the local antenna coordinate system, i.e., after
  applying the `orientation`. If no orientation vector is given, these angles are identical to the input
  elevation angles. Size: `nullptr` or `[n_out, n_ang]`

- `arma::Mat<dtype> ***gamma** = nullptr` (optional output)<br>
  Polarization rotation angles in [rad]. Size: `nullptr` or `[n_out, n_ang]`


## Example:
```
double pi = arma::datum::pi;

// Directional antenna, pointing east
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);

arma::mat azimuth = {0.0, 0.5 * pi, -0.5 * pi, pi};     // Azimuth angles: East, North, South, West
arma::mat elevation(1, azimuth.n_elem);                 // Initialize to 0
arma::mat V_re, V_im, H_re, H_im;                       // Output variables (uninitialized)
ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im);
V_re.print();
```
MD!*/

// ARRAYANT METHOD : Interpolation
template <typename dtype>
void quadriga_lib::arrayant<dtype>::interpolate(const arma::Mat<dtype> *azimuth,
                                                const arma::Mat<dtype> *elevation,
                                                arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
                                                arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
                                                arma::uvec i_element,
                                                const arma::Cube<dtype> *orientation,
                                                const arma::Mat<dtype> *element_pos_i,
                                                arma::Mat<dtype> *dist,
                                                arma::Mat<dtype> *azimuth_loc, arma::Mat<dtype> *elevation_loc,
                                                arma::Mat<dtype> *gamma) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (azimuth == nullptr)
        throw std::invalid_argument("Input 'azimuth' cannot be NULL");

    if (elevation == nullptr)
        throw std::invalid_argument("Input 'elevation' cannot be NULL");

    if (V_re == nullptr || V_im == nullptr || H_re == nullptr || H_im == nullptr)
        throw std::invalid_argument("Outputs 'V_re', 'V_im', 'H_re', 'H_im' cannot be NULL");

    arma::uword n_out = azimuth->n_rows;
    arma::uword n_ang = azimuth->n_cols;

    if (elevation->n_rows != n_out || elevation->n_cols != n_ang)
        throw std::invalid_argument("Sizes of 'azimuth' and 'elevation' do not match.");

    arma::uword n_elements = e_theta_re.n_slices;
    if (i_element.n_elem == 0)
        i_element = arma::regspace<arma::uvec>(0, n_elements - 1);
    else if (arma::any(i_element >= n_elements))
        throw std::invalid_argument("Element indices 'i_element' cannot exceed the array antenna size.");

    if (n_out != 1 && n_out != i_element.n_elem)
        throw std::invalid_argument("Number of requested outputs does not match the number of elements.");

    // Process orientation
    n_out = i_element.n_elem;
    arma::Cube<dtype> orientation_empty = arma::Cube<dtype>(3, 1, 1);
    const arma::Cube<dtype> *orientation_local = (orientation == nullptr || orientation->n_elem == 0) ? &orientation_empty : orientation;
    if (orientation != nullptr)
    {
        arma::uword o1 = orientation_local->n_rows,
                    o2 = orientation_local->n_cols,
                    o3 = orientation_local->n_slices;

        if (o1 != 3)
            throw std::invalid_argument("Input 'orientation' must have 3 elements on the first dimension.");
        else if (o2 != 1 && o2 != n_out)
            throw std::invalid_argument("Input 'orientation' must have 1 or 'n_elements' elements on the second dimension.");
        else if (o3 != 1 && o3 != n_ang)
            throw std::invalid_argument("Input 'orientation' must have 1 or 'n_ang' elements on the third dimension.");
    }

    // Process alternative element positions
    arma::Mat<dtype> element_pos_local(3, n_out);
    if (element_pos_i != nullptr && element_pos_i->n_elem != 0)
    {
        if (element_pos_i->n_rows != 3 || element_pos_i->n_cols != n_out)
            throw std::invalid_argument("Alternative element positions 'element_pos_i' must have 3 rows and 'n_out' columns.");

        const dtype *ptrI = element_pos_i->memptr();
        dtype *ptrO = element_pos_local.memptr();
        std::memcpy(ptrO, ptrI, 3 * n_out * sizeof(dtype));
    }
    else if (element_pos.n_elem != 0)
    {
        const dtype *ptrI = element_pos.memptr();
        dtype *ptrO = element_pos_local.memptr();
        for (arma::uword i = 0; i < n_out; ++i)
            std::memcpy(&ptrO[3 * i], &ptrI[3 * i_element[i]], 3 * sizeof(dtype));
    }

    // Resize output variables
    if (V_re->n_rows != n_out || V_re->n_cols != n_ang)
        V_re->set_size(n_out, n_ang);
    if (V_im->n_rows != n_out || V_im->n_cols != n_ang)
        V_im->set_size(n_out, n_ang);
    if (H_re->n_rows != n_out || H_re->n_cols != n_ang)
        H_re->set_size(n_out, n_ang);
    if (H_im->n_rows != n_out || H_im->n_cols != n_ang)
        H_im->set_size(n_out, n_ang);

    if (dist != nullptr && (dist->n_rows != n_out || dist->n_cols != n_ang))
        dist->set_size(n_out, n_ang);
    if (azimuth_loc != nullptr && (azimuth_loc->n_rows != n_out || azimuth_loc->n_cols != n_ang))
        azimuth_loc->set_size(n_out, n_ang);
    if (elevation_loc != nullptr && (elevation_loc->n_rows != n_out || elevation_loc->n_cols != n_ang))
        elevation_loc->set_size(n_out, n_ang);
    if (gamma != nullptr && (gamma->n_rows != n_out || gamma->n_cols != n_ang))
        gamma->set_size(n_out, n_ang);

    // Convert element index format
    arma::u32_vec i_element_int(n_out); // 1-based
    {
        unsigned *pi = i_element_int.memptr();
        arma::uword *pu = i_element.memptr();
        for (arma::uword i = 0; i < n_out; ++i)
            pi[i] = (unsigned)pu[i] + 1;
    }

    // Call private library function
    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, azimuth, elevation,
                            &i_element_int, orientation_local, &element_pos_local,
                            V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma);
}

/*!MD
# .qdant_write
Write array antenna object and layout to QDANT file

## Description:
- This function writes array antenna patterns and their layout into the QuaDRiGa array antenna exchange
  format (QDANT), an XML-based file format
- Multiple array antennas can be stored in the same file using the `id` parameter.
- If writing to an exisiting file without specifying an `id`, the data gests appended at the end.
  The output `id_in_file` identifies the location inside the file.
- An optional storage `layout` can be provided to organize data inside the file.
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
unsigned quadriga_lib::arrayant<dtype>::qdant_write(
                std::string fn,
                unsigned id = 0,
                arma::u32_mat layout = {}) const;
```

## Arguments:
- `std::string **fn**` (input)<br>
  Filename of the QDANT file to write the antenna pattern data. Cannot be empty.

- `unsigned **id** = 0` (optional input)<br>
  ID of the antenna to write into the file. If not provided or set to `0`, the antenna pattern is appended with a new ID equal to the maximum existing ID in the file plus one.

- `arma::u32_mat **layout** = {}` (optional input)<br>
  Layout specifying the organization of multiple antenna elements inside the file. This matrix must only contain element IDs present within the file. Default: empty matrix `{}`.

## Returns:
- `unsigned`<br>
  Returns the ID assigned to the antenna pattern within the file after writing.

## Example:
```
quadriga_lib::arrayant<double> ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
unsigned ant_id = ant.qdant_write("antenna_data.qdant");
```

## See also:
- [[arrayant]]
- <a href="#qdant_read">qdant_read</a>
- QuaDRiGa Array Antenna Exchange Format  (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

// ARRAYANT : Write to QDANT file
template <typename dtype>
unsigned quadriga_lib::arrayant<dtype>::qdant_write(std::string fn, unsigned id, arma::u32_mat layout) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    unsigned id_in_file = 0;
    error_message = qd_arrayant_qdant_write(fn, id, &name, &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                                            &azimuth_grid, &elevation_grid, &element_pos,
                                            &coupling_re, &coupling_im, &center_frequency, &layout,
                                            &id_in_file);

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    return id_in_file;
}

/*!MD
# .remove_zeros
Remove zeros from antenna pattern data

## Description:
- This function removes zeros from the antenna pattern data, altering its size accordingly.
- If called without an argument, the function modifies the antenna array properties in place.
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant<dtype>::remove_zeros(arrayant<dtype> *output = nullptr);
```

## Arguments:
- `arrayant<dtype> ***output** = nullptr` (optional output)<br>
  Pointer to an antenna array object where the modified pattern data is should be written to. If set
  to `nullptr` (default), the modifications are applied directly to the calling antenna object.

## Example:
```
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
ant.remove_zeros(); // Modifies ant in-place
```
MD!*/

// Remove zeros from the pattern
template <typename dtype>
void quadriga_lib::arrayant<dtype>::remove_zeros(quadriga_lib::arrayant<dtype> *output)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (output == nullptr && read_only)
    {
        error_message = "Cannot update read-only array antenna object inplace.";
        throw std::invalid_argument(error_message.c_str());
    }

    unsigned long long n_el = e_theta_re.n_rows;
    unsigned long long n_az = e_theta_re.n_cols;
    unsigned long long n_slices = e_theta_re.n_slices;

    // Calculate the power pattern
    arma::Mat<dtype> pow(n_el, n_az);
    dtype *pp = pow.memptr();
    dtype limit = dtype(1.0e-12), pi_half = dtype(arma::datum::pi / 2.0);

    for (auto is = 0ULL; is < n_slices; ++is)
    {
        dtype *pa = e_theta_re.slice_memptr(is), *pb = e_theta_im.slice_memptr(is),
              *pc = e_phi_re.slice_memptr(is), *pd = e_phi_im.slice_memptr(is);
        for (auto j = 0ULL; j < n_el * n_az; ++j)
            pp[j] += pa[j] * pa[j] + pb[j] * pb[j] + pc[j] * pc[j] + pd[j] * pd[j];
    }

    arma::Row<dtype> az_sum = arma::sum(pow, 0);
    arma::Col<dtype> el_sum = arma::sum(pow, 1);
    pow.reset();

    // Determine which parts of the pattern should be kept
    arma::uvec keep_az(n_az), keep_el(n_el);
    unsigned long long *keep_az_ptr = keep_az.memptr();
    unsigned long long *keep_el_ptr = keep_el.memptr();

    pp = az_sum.memptr();
    dtype *grid_ptr = azimuth_grid.memptr();
    for (auto ia = 0ULL; ia < n_az; ++ia)
    {
        dtype p = ia == 0 ? *grid_ptr + grid_ptr[n_az - 1] : grid_ptr[ia] - grid_ptr[ia - 1];
        dtype n = ia == n_az - 1 ? *grid_ptr + grid_ptr[ia] : grid_ptr[ia + 1] - grid_ptr[ia];

        unsigned long long ip = ia == 0ULL ? n_az - 1ULL : ia - 1ULL;
        unsigned long long in = ia == n_az - 1ULL ? 0ULL : ia + 1ULL;

        if (pp[ia] > limit)
            keep_az_ptr[ip] = n > 0.001 * p && p > 1.5 * n ? keep_az_ptr[ip] : 1,
            keep_az_ptr[ia] = 1,
            keep_az_ptr[in] = p > 0.001 * n && n > 1.5 * p ? keep_az_ptr[in] : 1;
    }
    az_sum.reset();

    pp = el_sum.memptr(), grid_ptr = elevation_grid.memptr();

    for (auto ie = 0ULL; ie < n_el; ++ie)
    {
        dtype p = ie == 0 ? *grid_ptr + pi_half : grid_ptr[ie] - grid_ptr[ie - 1];
        dtype n = ie == n_el - 1 ? pi_half - grid_ptr[ie] : grid_ptr[ie + 1] - grid_ptr[ie];

        unsigned long long ip = ie == 0ULL ? 0ULL : ie - 1ULL;
        unsigned long long in = ie == n_el - 1ULL ? n_el - 1ULL : ie + 1ULL;

        if (pp[ie] > limit)
            keep_el_ptr[ip] = n > 0.001 * p && p > 1.5 * n ? keep_el_ptr[ip] : 1ULL,
            keep_el_ptr[ie] = 1,
            keep_el_ptr[in] = p > 0.001 * n && n > 1.5 * p ? keep_el_ptr[in] : 1ULL;
    }
    el_sum.reset();

    // Copy the relevant pattern data
    unsigned long long n_az_new = arma::sum(keep_az), n_el_new = arma::sum(keep_el);

    for (int p = 0; p < 4; ++p)
    {
        dtype *data = nullptr, *ptrO = nullptr, *ptrI = nullptr;

        if (output == nullptr)
            data = new dtype[n_el_new * n_az_new * n_slices];
        else if (p == 0)
        {
            if (output->e_theta_re.n_rows != n_el_new || output->e_theta_re.n_cols != n_az_new || output->e_theta_re.n_slices != n_slices)
                output->e_theta_re.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_theta_re.memptr();
        }
        else if (p == 1)
        {
            if (output->e_theta_im.n_rows != n_el_new || output->e_theta_im.n_cols != n_az_new || output->e_theta_im.n_slices != n_slices)
                output->e_theta_im.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_theta_im.memptr();
        }
        else if (p == 2)
        {
            if (output->e_phi_re.n_rows != n_el_new || output->e_phi_re.n_cols != n_az_new || output->e_phi_re.n_slices != n_slices)
                output->e_phi_re.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_phi_re.memptr();
        }
        else if (p == 3)
        {
            if (output->e_phi_im.n_rows != n_el_new || output->e_phi_im.n_cols != n_az_new || output->e_phi_im.n_slices != n_slices)
                output->e_phi_im.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_phi_im.memptr();
        }

        ptrO = data;
        for (auto is = 0ULL; is < n_slices; ++is)
        {
            if (p == 0)
                ptrI = e_theta_re.slice_memptr(is);
            else if (p == 1)
                ptrI = e_theta_im.slice_memptr(is);
            else if (p == 2)
                ptrI = e_phi_re.slice_memptr(is);
            else if (p == 3)
                ptrI = e_phi_im.slice_memptr(is);

            for (auto ia = 0ULL; ia < n_az; ++ia)
            {
                bool keep_az = keep_az_ptr[ia] == 1;
                for (auto ie = 0ULL; ie < n_el; ++ie)
                    if (keep_az && keep_el_ptr[ie] == 1ULL)
                        *ptrO++ = ptrI[ia * n_el + ie];
            }
        }

        if (output == nullptr)
        {
            if (p == 0)
                e_theta_re.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_theta_re.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 1)
                e_theta_im.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_theta_im.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 2)
                e_phi_re.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_phi_re.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 3)
                e_phi_im.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_phi_im.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));

            delete[] data;
        }
    }

    // Copy the new azimuth grid
    dtype *azimuth_grid_new = new dtype[n_az_new];
    pp = azimuth_grid_new;
    for (auto ia = 0ULL; ia < n_az; ++ia)
        if (keep_az_ptr[ia] == 1)
            *pp++ = azimuth_grid[ia];
    if (output == nullptr)
        azimuth_grid.set_size(n_az_new), pp = azimuth_grid.memptr();
    else if (output->azimuth_grid.n_elem != n_az_new)
        output->azimuth_grid.set_size(n_az_new), pp = output->azimuth_grid.memptr();
    else
        pp = output->azimuth_grid.memptr();
    std::memcpy(pp, azimuth_grid_new, n_az_new * sizeof(dtype));
    delete[] azimuth_grid_new;

    // Copy the new elevation grid
    dtype *elevation_grid_new = new dtype[n_el_new];
    pp = elevation_grid_new;
    for (auto ie = 0ULL; ie < n_el; ++ie)
        if (keep_el_ptr[ie] == 1)
            *pp++ = elevation_grid[ie];
    if (output == nullptr)
        elevation_grid.set_size(n_el_new), pp = elevation_grid.memptr();
    else if (output->elevation_grid.n_elem != n_el_new)
        output->elevation_grid.set_size(n_el_new), pp = output->elevation_grid.memptr();
    else
        pp = output->elevation_grid.memptr();
    std::memcpy(pp, elevation_grid_new, n_el_new * sizeof(dtype));
    delete[] elevation_grid_new;

    // Copy the unchanged data
    if (output != nullptr)
    {
        if (!element_pos.empty())
        {
            if (output->element_pos.n_rows != 3 || output->element_pos.n_cols != n_slices)
                output->element_pos.set_size(3, n_slices);
            std::memcpy(output->element_pos.memptr(), element_pos.memptr(), 3 * n_slices * sizeof(dtype));
        }

        if (!coupling_re.empty())
        {
            if (output->coupling_re.n_rows != n_slices || output->coupling_re.n_cols != coupling_re.n_cols)
                output->coupling_re.set_size(n_slices, coupling_re.n_cols);
            std::memcpy(output->coupling_re.memptr(), coupling_re.memptr(), n_slices * coupling_re.n_cols * sizeof(dtype));
        }

        if (!coupling_im.empty())
        {
            if (output->coupling_im.n_rows != n_slices || output->coupling_im.n_cols != coupling_im.n_cols)
                output->coupling_im.set_size(n_slices, coupling_im.n_cols);
            std::memcpy(output->coupling_im.memptr(), coupling_im.memptr(), n_slices * coupling_im.n_cols * sizeof(dtype));
        }

        output->center_frequency = center_frequency;
        output->validate();
    }
}

// ARRAYANT METHOD : Change the size of an arrayant, without explicitly preserving data
template <typename dtype>
void quadriga_lib::arrayant<dtype>::reset()
{
    if (read_only)
    {
        std::string error_message = "Cannot change size of read-only array antenna object.";
        throw std::invalid_argument(error_message.c_str());
    }

    if (azimuth_grid.n_elem != 0)
        azimuth_grid.reset();

    if (elevation_grid.n_elem != 0)
        elevation_grid.reset();

    if (e_theta_re.n_elem != 0)
        e_theta_re.reset();

    if (e_theta_im.n_elem != 0)
        e_theta_im.reset();

    if (e_phi_re.n_elem != 0)
        e_phi_re.reset();

    if (e_phi_im.n_elem != 0)
        e_phi_im.reset();

    if (element_pos.n_elem != 0)
        element_pos.reset();

    if (coupling_re.n_elem != 0)
        coupling_re.reset();

    if (coupling_im.n_elem != 0)
        coupling_im.reset();

    center_frequency = dtype(299792458.0);
}

/*!MD
# .rotate_pattern
Adjust orientation of antenna patterns

## Description:

* Adjusts the orientation of antenna radiation patterns by performing precise rotations around the
  three principal axes (x, y, z) of the local Cartesian coordinate system (Euler rotations)
* Transforms both uniformly and non-uniformly sampled antenna patterns, useful for precise adjustments
  in antennas like parabolic antennas with small apertures.
* Member function of [[arrayant]]
* Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant<dtype>::rotate_pattern(
                dtype x_deg = 0.0,
                dtype y_deg = 0.0,
                dtype z_deg = 0.0,
                unsigned usage = 0,
                unsigned element = -1,
                arrayant<dtype> *output = nullptr);
```

## Arguments:
- `dtype **x_deg** = 0.0` (optional input)<br>
  Rotation angle around the x-axis (bank angle), specified in degrees. Default: `0.0`

- `dtype **y_deg** = 0.0` (optional input)<br>
  Rotation angle around the y-axis (tilt angle), specified in degrees. Default: `0.0`

- `dtype **z_deg** = 0.0` (optional input)<br>
  Rotation angle around the z-axis (heading angle), specified in degrees. Default: `0.0`

- `unsigned **usage** = 0` (optional input)<br>
  Rotation usage model, specifying which components to rotate: (`0`): Rotate both pattern and polarization,
  (`1`): Rotate only pattern, (`2`): Rotate only polarization, (`3`): Rotate both pattern and polarization without adjusting the grid

- `unsigned **element** = -1` (optional input)<br>
  Index of the antenna element (0-based) to rotate. Default (`-1`) applies rotation to all elements.

- `arrayant<dtype> **output** = nullptr` (optional output)<br>
  Pointer to an antenna array object to store the modified pattern data. If `nullptr` (default), modifications are applied directly to the calling antenna object.

## Example:
```
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
ant.rotate_pattern(0.0, 0.0, 45.0);
```
MD!*/

// Rotating antenna patterns (adjusts sampling grid if needed, e.g. for parabolic antennas)
template <typename dtype>
void quadriga_lib::arrayant<dtype>::rotate_pattern(dtype x_deg, dtype y_deg, dtype z_deg,
                                                   unsigned usage, unsigned element,
                                                   quadriga_lib::arrayant<dtype> *output)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    bool use_all_elements = false;
    if (usage > 3)
        error_message = "Input parameter 'usage' must be 0, 1, 2 or 3.";
    if (element == unsigned(-1) || e_theta_re.n_slices == 1)
        use_all_elements = true;
    if (element != unsigned(-1) && element >= e_theta_re.n_slices)
        error_message = "Input parameter 'element' out of bound.";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (output == nullptr && read_only)
    {
        error_message = "Cannot update read-only array antenna object inplace.";
        throw std::invalid_argument(error_message.c_str());
    }

    // Set element indices
    arma::Col<unsigned> i_element(1);
    if (use_all_elements)
        i_element = arma::regspace<arma::Col<unsigned>>(1, unsigned(e_theta_re.n_slices));
    else
        i_element.at(0) = element + 1;

    // Extract the element positions
    arma::Mat<dtype> element_pos_update;
    if (element_pos.n_elem == 0)
        element_pos_update.zeros(3, e_theta_re.n_slices);
    else
        element_pos_update = element_pos;

    unsigned long long n_el = e_theta_re.n_rows;
    unsigned long long n_az = e_theta_re.n_cols;
    unsigned long long n_out = i_element.n_elem;
    unsigned long long n_ang = n_el * n_az;

    dtype tau = dtype(arma::datum::tau), pi_half = dtype(arma::datum::pi / 2.0), pi = dtype(arma::datum::pi),
          deg2rad = dtype(arma::datum::pi / 180.0), limit = dtype(1.0e-6);

    // Calculate the coverage range for the angle sampling grid
    dtype zero = dtype(0.0), az_step_min = dtype(1e38), az_step_max = zero, el_step_min = az_step_min,
          el_step_max = zero, step = zero;
    dtype *ptr = azimuth_grid.memptr();
    for (auto i = 0ULL; i < n_az; ++i)
    {
        step = i == 0 ? *ptr - ptr[n_az - 1] + tau : ptr[i] - ptr[i - 1];
        az_step_min = step < az_step_min && step > limit ? step : az_step_min;
        az_step_max = step > az_step_max ? step : az_step_max;
    }
    ptr = elevation_grid.memptr();
    for (auto i = 0ULL; i <= n_el; ++i)
    {
        if (i == 0)
            step = *ptr + pi_half;
        else if (i == n_el)
            step = pi_half - ptr[i - 1];
        else
            step = ptr[i] - ptr[i - 1];

        el_step_min = step < el_step_min && step > limit ? step : el_step_min;
        el_step_max = step > el_step_max ? step : el_step_max;
    }
    step = pi_half - ptr[n_el - 1];
    el_step_min = step < el_step_min && step > limit ? step : el_step_min;
    el_step_max = step > el_step_max ? step : el_step_max;
    step = el_step_min < az_step_min ? el_step_min : az_step_min;

    bool update_grid = (usage == 0 || usage == 1) && (az_step_max - az_step_min > limit ||
                                                      el_step_max - el_step_min > limit ||
                                                      az_step_min - el_step_min > limit ||
                                                      n_az == 1 || n_el == 1)
                           ? true
                           : false;

    // Obtain interpolation angles
    arma::Col<dtype> azimuth_grid_update, elevation_grid_update;
    if (update_grid)
    {
        if (output == nullptr && !use_all_elements)
        {
            error_message = "Update of sampling grid cannot be done for single elements of an array antenna!";
            throw std::invalid_argument(error_message.c_str());
        }

        if (step > dtype(0.0017)) // >= 0.1 degree --> Use entire sphere
        {
            unsigned long long N = (unsigned long long)std::round(pi / step);
            N = N < 4ULL ? 4ULL : N;

            azimuth_grid_update = arma::linspace<arma::Col<dtype>>(-pi, pi, 2ULL * N + 1ULL);

            if (std::abs(x_deg) < limit && std::abs(y_deg) < limit)
                elevation_grid_update = elevation_grid;
            else
                elevation_grid_update = arma::linspace<arma::Col<dtype>>(-pi_half + step, pi_half - step, N - 1ULL);
        }
        else // step < 0.1 degree --> Use subsampled sphere
        {
            auto ant = quadriga_lib::generate_arrayant_omni<dtype>();
            ant.e_theta_re.zeros();
            arma::Col<unsigned> has_az(361), has_el(181);

            // Select azimuth angles
            dtype *p = azimuth_grid.memptr(), *q = ant.azimuth_grid.memptr();
            unsigned *ra = has_az.memptr();
            for (unsigned i = 0; i < 360; ++i)
            {
                while (p != azimuth_grid.end() && *p <= q[i + 1])
                {
                    ra[i] = *p >= q[i] && *p <= q[i + 1] ? 1 : ra[i];
                    p++;
                }
                ra[i + 1] = ra[i];
                if (p == azimuth_grid.end())
                    break;
            }

            // Select elevation angles
            p = elevation_grid.memptr(), q = ant.elevation_grid.memptr();
            unsigned *re = has_el.memptr();
            for (unsigned i = 0; i < 180; ++i)
            {
                while (p != elevation_grid.end() && *p <= q[i + 1])
                {
                    re[i] = *p >= q[i] && *p <= q[i + 1] ? 1 : re[i];
                    p++;
                }
                re[i + 1] = re[i];
                if (p == elevation_grid.end())
                    break;
            }

            // Set interpolation target
            p = ant.e_theta_re.memptr();
            for (auto i = 0ULL; i < ant.e_theta_re.n_elem; ++i)
                p[i] = ra[i / 181] == 1 && re[i % 181] == 1 ? 1.0 : 0.0;

            // Find target area for interpolation
            ant.rotate_pattern(x_deg, y_deg, z_deg, 1);
            ant.remove_zeros();

            // Subdivide target grid
            dtype sdiv = std::fmod(deg2rad, step) < 0.1 * step ? std::floor(deg2rad / step) : std::ceil(deg2rad / step);
            dtype stp = deg2rad / sdiv;
            unsigned long long ndiv = (unsigned long long)sdiv;

            bool wrap = ant.azimuth_grid.at(ant.azimuth_grid.n_elem - 1) + stp > pi;
            unsigned long long N = wrap ? ndiv * (ant.azimuth_grid.n_elem - 1) + 1 : ndiv * ant.azimuth_grid.n_elem;
            azimuth_grid_update.set_size(N);
            p = azimuth_grid_update.memptr(), q = ant.azimuth_grid.memptr();
            for (auto i = 0ULL; i < ant.azimuth_grid.n_elem; ++i)
                if (wrap && i == ant.azimuth_grid.n_elem - 1ULL)
                    *p = q[i];
                else
                    for (auto j = 0ULL; j < ndiv; ++j)
                        *p++ = q[i] + dtype(j) * stp;

            wrap = ant.elevation_grid.at(ant.elevation_grid.n_elem - 1) + stp > pi_half;
            N = wrap ? ndiv * (ant.elevation_grid.n_elem - 1) + 1 : ndiv * ant.elevation_grid.n_elem;
            elevation_grid_update.set_size(N);
            p = elevation_grid_update.memptr(), q = ant.elevation_grid.memptr();
            for (auto i = 0ULL; i < ant.elevation_grid.n_elem; ++i)
                if (wrap && i == ant.elevation_grid.n_elem - 1)
                    *p = q[i];
                else
                    for (auto j = 0ULL; j < ndiv; ++j)
                        *p++ = q[i] + dtype(j) * stp;
        }

        n_az = azimuth_grid_update.n_elem;
        n_el = elevation_grid_update.n_elem;
        n_ang = n_az * n_el;
    }
    else
    {
        azimuth_grid_update = azimuth_grid;
        elevation_grid_update = elevation_grid;
    }

    // Initiate output variables

    arma::Mat<dtype> V_re(n_out, n_ang, arma::fill::none), V_im(n_out, n_ang, arma::fill::none);
    arma::Mat<dtype> H_re(n_out, n_ang, arma::fill::none), H_im(n_out, n_ang, arma::fill::none);
    arma::Mat<dtype> azimuth_loc, elevation_loc, gamma, EMPTY;

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr(),
          *p_phi = azimuth_grid_update.memptr(), *p_theta = elevation_grid_update.memptr();

    for (auto ia = 0ULL; ia < n_az; ++ia)
        for (auto ie = 0ULL; ie < n_el; ++ie)
            *p_azimuth++ = p_phi[ia], *p_elevation++ = p_theta[ie];

    // Calculate rotation matrix
    arma::Cube<dtype> orientation(3, 1, 1);
    orientation.at(0) = x_deg * deg2rad;
    orientation.at(1) = -y_deg * deg2rad;
    orientation.at(2) = z_deg * deg2rad;

    // Set antenna orientation
    arma::Cube<dtype> R = quadriga_lib::calc_rotation_matrix(orientation, true);

    if (usage == 1)
        azimuth_loc.set_size(n_out, n_ang), elevation_loc.set_size(n_out, n_ang);
    else if (usage == 2)
        gamma.set_size(n_out, n_ang);

    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
                            &i_element, &orientation, &element_pos_update,
                            &V_re, &V_im, &H_re, &H_im, &EMPTY, &azimuth_loc, &elevation_loc, &gamma);
    azimuth.reset(), elevation.reset();

    orientation.zeros();
    if (usage == 1) // Only return interpolated pattern (ignore polarization)
        qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                                &azimuth_grid, &elevation_grid, &azimuth_loc, &elevation_loc,
                                &i_element, &orientation, &element_pos_update,
                                &V_re, &V_im, &H_re, &H_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);
    azimuth_loc.reset(), elevation_loc.reset();

    // Adjust size of output, if needed
    if (output != nullptr)
        output->set_size(n_el, n_az, n_out, n_out);

    if (usage == 2) // Only adjust the polarization
    {
        gamma = gamma.t();
        for (auto i = 0ULL; i < n_out; ++i)
        {
            dtype *p_gamma = gamma.colptr(i);
            dtype *p_theta_re = use_all_elements ? e_theta_re.slice_memptr(i) : e_theta_re.slice_memptr(element);
            dtype *p_theta_im = use_all_elements ? e_theta_im.slice_memptr(i) : e_theta_im.slice_memptr(element);
            dtype *p_phi_re = use_all_elements ? e_phi_re.slice_memptr(i) : e_phi_re.slice_memptr(element);
            dtype *p_phi_im = use_all_elements ? e_phi_im.slice_memptr(i) : e_phi_im.slice_memptr(element);

            dtype *q_theta_re = p_theta_re, *q_theta_im = p_theta_im, *q_phi_re = p_phi_re, *q_phi_im = p_phi_im;
            if (output != nullptr)
                q_theta_re = use_all_elements ? output->e_theta_re.slice_memptr(i) : output->e_theta_re.memptr(),
                q_theta_im = use_all_elements ? output->e_theta_im.slice_memptr(i) : output->e_theta_im.memptr(),
                q_phi_re = use_all_elements ? output->e_phi_re.slice_memptr(i) : output->e_phi_re.memptr(),
                q_phi_im = use_all_elements ? output->e_phi_im.slice_memptr(i) : output->e_phi_im.memptr();

            for (auto j = 0ULL; j < n_ang; ++j)
            {
                dtype sin_gamma = std::sin(p_gamma[j]), cos_gamma = std::cos(p_gamma[j]);
                dtype tmp = sin_gamma * p_theta_re[j];
                q_theta_re[j] = cos_gamma * p_theta_re[j] - sin_gamma * p_phi_re[j];
                q_phi_re[j] = tmp + cos_gamma * p_phi_re[j];
                tmp = sin_gamma * p_theta_im[j];
                q_theta_im[j] = cos_gamma * p_theta_im[j] - sin_gamma * p_phi_im[j];
                q_phi_im[j] = tmp + cos_gamma * p_phi_im[j];
            }
        }
        gamma.reset();

        // Copy unchanged data
        if (output != nullptr)
        {
            dtype *ptrI = use_all_elements ? element_pos_update.memptr() : element_pos_update.colptr(element);
            dtype *ptrO = output->element_pos.memptr();
            std::memcpy(ptrO, ptrI, 3 * n_out * sizeof(dtype));

            ptrI = azimuth_grid_update.memptr();
            ptrO = output->azimuth_grid.memptr();
            std::memcpy(ptrO, ptrI, n_az * sizeof(dtype));

            ptrI = elevation_grid_update.memptr();
            ptrO = output->elevation_grid.memptr();
            std::memcpy(ptrO, ptrI, n_el * sizeof(dtype));
        }
    }
    else // Usage 0, 1 or 3
    {
        gamma.reset();
        if (n_out > 1) // Transpose the interpolated data
            V_re = V_re.t(), V_im = V_im.t(), H_re = H_re.t(), H_im = H_im.t();

        if (output == nullptr && update_grid) // Set new array size
            set_size(n_el, n_az, n_out, n_out);

        dtype *ptrI, *ptrO;

        ptrI = V_re.memptr();
        ptrO = use_all_elements ? e_theta_re.memptr() : e_theta_re.slice_memptr(element);
        ptrO = output == nullptr ? ptrO : output->e_theta_re.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        V_re.reset();

        ptrI = V_im.memptr();
        ptrO = use_all_elements ? e_theta_im.memptr() : e_theta_im.slice_memptr(element);
        ptrO = output == nullptr ? ptrO : output->e_theta_im.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        V_im.reset();

        ptrI = H_re.memptr();
        ptrO = use_all_elements ? e_phi_re.memptr() : e_phi_re.slice_memptr(element);
        ptrO = output == nullptr ? ptrO : output->e_phi_re.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        H_re.reset();

        ptrI = H_im.memptr();
        ptrO = use_all_elements ? e_phi_im.memptr() : e_phi_im.slice_memptr(element);
        ptrO = output == nullptr ? ptrO : output->e_phi_im.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        H_im.reset();

        ptrI = azimuth_grid_update.memptr();
        ptrO = output == nullptr ? azimuth_grid.memptr() : output->azimuth_grid.memptr();
        std::memcpy(ptrO, ptrI, n_az * sizeof(dtype));

        ptrI = elevation_grid_update.memptr();
        ptrO = output == nullptr ? elevation_grid.memptr() : output->elevation_grid.memptr();
        std::memcpy(ptrO, ptrI, n_el * sizeof(dtype));

        ptrI = element_pos_update.memptr();
        ptrO = use_all_elements ? element_pos.memptr() : element_pos.colptr(element);
        ptrO = output == nullptr ? ptrO : output->element_pos.memptr();
        dtype *R_ptr = R.memptr();
        for (auto i = 0ULL; i < i_element.n_elem; ++i)
        {
            unsigned j = 3 * (i_element.at(i) - 1);
            unsigned k = use_all_elements ? j : 0;
            dtype a = R_ptr[0] * ptrI[j] + R_ptr[3] * ptrI[j + 1] + R_ptr[6] * ptrI[j + 2];
            dtype b = R_ptr[1] * ptrI[j] + R_ptr[4] * ptrI[j + 1] + R_ptr[7] * ptrI[j + 2];
            dtype c = R_ptr[2] * ptrI[j] + R_ptr[5] * ptrI[j + 1] + R_ptr[8] * ptrI[j + 2];
            ptrO[k] = a, ptrO[k + 1] = b, ptrO[k + 2] = c;
        }

        if (output == nullptr && update_grid)
            remove_zeros();
        else if (output != nullptr && update_grid)
            output->remove_zeros();
    }
}

/*!MD
# .set_size
Change size of antenna array object

## Description:
- Changes the size of an antenna array (`arrayant`) without explicitly preserving existing data.
- Resets `element_pos` to zero and sets `coupling_re` and `coupling_im` to identity matrices.
- Other properties may contain undefined or garbage data after resizing
- Size update is performed only if the existing size differs from the specified new size
- Function returns an error if the antenna object is marked as read-only
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant<dtype>::set_size(
                arma::uword n_elevation,
                arma::uword n_azimuth,
                arma::uword n_elements,
                arma::uword n_ports);
```

## Arguments:
- `arma::uword **n_elevation**` (input)<br>
  Number of elevation angles to resize to.

- `arma::uword **n_azimuth**` (input)<br>
  Number of azimuth angles to resize to.

- `arma::uword **n_elements**` (input)<br>
  Number of antenna elements in the array after resizing.

- `arma::uword **n_ports**` (input)<br>
  Number of ports (after coupling of elements) in the resized antenna array.

## Example:
```
quadriga_lib::arrayant<double> ant;
ant.set_size(180, 360, 4, 2);
```
MD!*/

// ARRAYANT METHOD : Change the size of an arrayant, without explicitly preserving data
template <typename dtype>
void quadriga_lib::arrayant<dtype>::set_size(arma::uword n_elevation, arma::uword n_azimuth,
                                             arma::uword n_elements, arma::uword n_ports)
{
    if (read_only)
    {
        std::string error_message = "Cannot change size of read-only array antenna object.";
        throw std::invalid_argument(error_message.c_str());
    }

    if (azimuth_grid.n_elem != n_azimuth)
        azimuth_grid.set_size(n_azimuth);

    if (elevation_grid.n_elem != n_elevation)
        elevation_grid.set_size(n_elevation);

    if (e_theta_re.n_rows != n_elevation || e_theta_re.n_cols != n_azimuth || e_theta_re.n_slices != n_elements)
        e_theta_re.set_size(n_elevation, n_azimuth, n_elements);

    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements)
        e_theta_im.set_size(n_elevation, n_azimuth, n_elements);

    if (e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements)
        e_phi_re.set_size(n_elevation, n_azimuth, n_elements);

    if (e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        e_phi_im.set_size(n_elevation, n_azimuth, n_elements);

    if (element_pos.n_rows != 3 || element_pos.n_cols != n_elements)
        element_pos.zeros(3, n_elements);

    if (coupling_re.n_rows != n_elements || coupling_re.n_cols != n_ports)
        coupling_re.eye(n_elements, n_ports);

    if (coupling_im.n_rows != n_elements || coupling_im.n_cols != n_ports)
        coupling_im.zeros(n_elements, n_ports);
}

/*!MD
# .is_valid
Validate integrity of antenna array object

## Description:
- Checks the integrity of an antenna array (`arrayant`) object.
- Returns an empty string if the antenna object is valid.
- Provides an error message describing any issue if the object is invalid.
- A quick integrity check can be performed for efficiency.
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
std::string quadriga_lib::arrayant<dtype>::is_valid(bool quick_check = true) const;
```

## Arguments:

- `bool **quick_check** = true` (optional input)<br>
  If set to `true` (default), performs a quick validation check. Setting it to `false` performs a more thorough validation.

## Returns:
- `std::string`<br>
  Returns an empty string (`""`) if the antenna object passes the integrity check; otherwise, returns an error message detailing the issue.

## Example:
```
quadriga_lib::arrayant<double> ant;
std::string result = ant.is_valid();
if(result.empty()) {
    std::cout << "Antenna array is valid." << std::endl;
} else {
    std::cout << "Error: " << result << std::endl;
}
```
MD!*/

// ARRAYANT METHOD : Validates correctness of the member functions
template <typename dtype>
std::string quadriga_lib::arrayant<dtype>::is_valid(bool quick_check) const
{

    // Assuming that the data has been validated before, we can quickly check if the
    // data pointers were updated. If not, we can assume that the data is still valid.
    if (quick_check)
        quick_check = check_ptr[0] != nullptr && check_ptr[1] != nullptr && check_ptr[2] != nullptr &&
                      check_ptr[3] != nullptr && check_ptr[4] != nullptr && check_ptr[5] != nullptr &&
                      check_ptr[0] == e_theta_re.memptr() &&
                      check_ptr[1] == e_theta_im.memptr() &&
                      check_ptr[2] == e_phi_re.memptr() &&
                      check_ptr[3] == e_phi_im.memptr() &&
                      check_ptr[4] == azimuth_grid.memptr() &&
                      check_ptr[5] == elevation_grid.memptr() &&
                      check_ptr[6] == element_pos.memptr() &&
                      check_ptr[7] == coupling_re.memptr() &&
                      check_ptr[8] == coupling_im.memptr();
    if (quick_check)
        return std::string("");

    // Perform a deep check
    if (e_theta_re.n_elem == 0ULL || e_theta_im.n_elem == 0ULL || e_phi_re.n_elem == 0ULL || e_phi_im.n_elem == 0ULL ||
        azimuth_grid.n_elem == 0ULL || elevation_grid.n_elem == 0ULL)
        return std::string("Missing data for any of: e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid");

    arma::uword n_elevation = e_theta_re.n_rows;
    arma::uword n_azimuth = e_theta_re.n_cols;
    arma::uword n_elements = e_theta_re.n_slices;

    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements)
        return std::string("Sizes of 'e_theta_re' and 'e_theta_im' do not match.");

    if (e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements)
        return std::string("Sizes of 'e_theta_re' and 'e_phi_re' do not match.");

    if (e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        return std::string("Sizes of 'e_theta_re' and 'e_phi_im' do not match.");

    if (azimuth_grid.n_elem != n_azimuth)
        return std::string("Number of elements in 'azimuth_grid' does not match number of columns in pattern data.");

    if (elevation_grid.n_elem != n_elevation)
        return std::string("Number of elements in 'elevation_grid' does not match number of rows in pattern data.");

    bool error_code = false;
    for (const dtype *val = azimuth_grid.begin(); val < azimuth_grid.end(); ++val)
        error_code = *val < dtype(-3.1415930) || *val > dtype(3.1415930) ? true : error_code;
    if (error_code)
        return std::string("Values of 'azimuth_grid' must be between -pi and pi (equivalent to -180 to 180 degree).");

    for (const dtype *val = elevation_grid.begin(); val < elevation_grid.end(); ++val)
        error_code = *val < dtype(-1.5707965) || *val > dtype(1.5707965) ? true : error_code;
    if (error_code)
        return std::string("Values of 'elevation_grid' must be between -pi/2 and pi/2 (equivalent to -90 to 90 degree).");

    if (!azimuth_grid.is_sorted())
        return std::string("Values of 'azimuth_grid' must be sorted in ascending order.");

    if (!elevation_grid.is_sorted())
        return std::string("Values of 'elevation_grid' must be sorted in ascending order.");

    if (!element_pos.empty() && (element_pos.n_rows != 3 || element_pos.n_cols != n_elements))
        return std::string("Size of 'element_pos' must be either empty or match [3, n_elements]");

    if (!coupling_re.empty() && coupling_re.n_rows != n_elements)
        return std::string("'Coupling' must be a matrix with rows equal to number of elements");

    if (coupling_re.empty() && !coupling_im.empty())
        return std::string("Imaginary part of coupling matrix (phase component) defined without real part (absolute component)");

    if (!coupling_im.empty() && (coupling_im.n_rows != n_elements || coupling_im.n_cols != coupling_re.n_cols))
        return std::string("'coupling_im' must be empty or its size must match 'coupling_re'");

    return std::string("");
}

// ARRAYANT METHOD : Validates correctness of the member functions and initializes element positions and coupling matrix
template <typename dtype>
std::string quadriga_lib::arrayant<dtype>::validate()
{
    std::string error_message = is_valid(false); // Deep check
    if (error_message.length() != 0)
        return error_message;

    unsigned long long n_elements = e_theta_re.n_slices;
    unsigned long long n_prt = coupling_re.empty() ? n_elements : coupling_re.n_cols;

    if (element_pos.empty())
        element_pos.zeros(3, n_elements);

    if (coupling_re.empty())
        coupling_re.eye(n_elements, n_elements), coupling_im.zeros(n_elements, n_elements);

    if (coupling_im.empty())
        coupling_im.zeros(n_elements, n_prt);

    // Set the data pointers for the quick check.
    check_ptr[0] = e_theta_re.memptr();
    check_ptr[1] = e_theta_im.memptr();
    check_ptr[2] = e_phi_re.memptr();
    check_ptr[3] = e_phi_im.memptr();
    check_ptr[4] = azimuth_grid.memptr();
    check_ptr[5] = elevation_grid.memptr();
    check_ptr[6] = element_pos.memptr();
    check_ptr[7] = coupling_re.memptr();
    check_ptr[8] = coupling_im.memptr();

    return std::string("");
}

// Instantiate templates
template class quadriga_lib::arrayant<float>;
template class quadriga_lib::arrayant<double>;
