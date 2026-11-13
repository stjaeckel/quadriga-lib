// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <stdexcept>
#include <cstring>   // std::memcopy
#include <algorithm> // std::replace

#include "quadriga_arrayant.hpp"
#include "quadriga_tools.hpp"
#include "quadriga_math.hpp"
#include "qd_arrayant_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Array antenna class
SECTION!*/

/*!MD
# arrayant<++>
Class for storing and manipulating array antenna models

- Represents a multi-element antenna array; each element has a position relative to the array phase-center
- Elements may be inter-coupled via a complex coupling matrix
- Field pattern cubes `e_theta_re/im`, `e_phi_re/im` must all be `[n_elevation, n_azimuth, n_elements]`
- `element_pos` is optional (empty = all elements at origin); `coupling_re/im` are optional (empty = identity); all
  functions give the same results for empty members as for zeros / identity, except the IRS array of
  [[get_channels_irs]], whose coupling holds the IRS weights and must be given. In a multi-frequency vector all
  entries use the same form (see [[arrayant_is_valid_multi]])
- Field values and coupling weights must be finite with magnitudes below about 1e18; larger values overflow in the
  interpolation and in the channel functions and give NaN results

## Attributes:<br>
| Attribute                         | Size                                   | Description                                                    |
| --------------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| `arma::Cube<dtype> e_theta_re`    | `[n_elevation, n_azimuth, n_elements]` | E-theta (vertical) field, real part                            |
| `arma::Cube<dtype> e_theta_im`    | `[n_elevation, n_azimuth, n_elements]` | E-theta (vertical) field, imaginary part                       |
| `arma::Cube<dtype> e_phi_re`      | `[n_elevation, n_azimuth, n_elements]` | E-phi (horizontal) field, real part                            |
| `arma::Cube<dtype> e_phi_im`      | `[n_elevation, n_azimuth, n_elements]` | E-phi (horizontal) field, imaginary part                       |
| `arma::Col<dtype> azimuth_grid`   | `[n_azimuth]`                          | Azimuth angles in rad, in [-pi, pi], strictly increasing       |
| `arma::Col<dtype> elevation_grid` | `[n_elevation]`                        | Elevation angles in rad, in [-pi/2, pi/2], strictly increasing |
| `arma::Mat<dtype> element_pos`    | `[3, n_elements]` or empty             | Element positions in local Cartesian coords                    |
| `arma::Mat<dtype> coupling_re`    | `[n_elements, n_ports]` or empty       | Coupling matrix, real part                                     |
| `arma::Mat<dtype> coupling_im`    | `[n_elements, n_ports]` or empty       | Coupling matrix, imaginary part                                |
| `dtype center_frequency`          | scalar                                 | Center frequency                                               |
| `std::string name`                | string                                 | Name of the array antenna object                               |

## Simple member functions:<br>
| Function         | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `.n_elevation()` | Number of elevation angles                                             |
| `.n_azimuth()`   | Number of azimuth angles                                               |
| `.n_elements()`  | Number of antenna elements                                             |
| `.n_ports()`     | Number of ports (columns of coupling matrix)                           |
| `.copy()`        | Returns a deep copy; a copy of a valid arrayant is marked as validated |
| `.reset()`       | Clears all data, resetting size to zero                                |
| `.is_valid()`    | Returns `""` if valid, or an error message string                      |

## Complex member functions:<br>
| Function                  | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| .[[calc_beamwidth_deg]]   | Calculate the beam width of an antenna element in degree   |
| .[[calc_directivity_dBi]] | Calculate the directivity in dBi of a single array element |
| .[[set_size]]             | Resize the arrayant to new dimensions                      |
| .[[is_valid]]             | Validate arrayant integrity                                |

Computations and file I/O on an arrayant are free functions, see section "Array antenna functions"
(e.g. [[arrayant_concat]], [[arrayant_combine_pattern]], [[arrayant_copy_element]], [[arrayant_rotate_pattern]],
[[qdant_write]]).
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
# .calc_beamwidth_deg
Calculate the beamwidth and pointing angles of array antenna elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in each direction than the antenna sampling grid)
- The width is that of the main lobe: starting at the maximum, the search goes outwards until the power drops below the
  threshold; side lobes above the threshold are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360° when
  the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°
- Ignores element coupling

## Declaration:
```
void calc_beamwidth_deg(arma::uword i_element,
    dtype threshold_dB = 3.0,
    dtype *beamwidth_az = nullptr,
    dtype *beamwidth_el = nullptr,
    dtype *az_point_ang = nullptr,
    dtype *el_point_ang = nullptr) const;
```

## Inputs:
- **`i_element`** — Element index; 0-based
- **`threshold_dB`** — Threshold in dB; 3 dB = FWHM

## Outputs:
- **`beamwidth_az`** — Azimuth beamwidth in degree
- **`beamwidth_el`** — Elevation beamwidth in degree
- **`az_point_ang`** — Azimuth pointing angle for the main beam in degree
- **`el_point_ang`** — Elevation pointing angle for the main beam in degree

## See also:
- .[[calc_directivity_dBi]] (directivity in dBi of a single array element)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant<dtype>::calc_beamwidth_deg(arma::uword i_element, dtype threshold_dB,
                                                       dtype *beamwidth_az, dtype *beamwidth_el,
                                                       dtype *az_point_ang, dtype *el_point_ang) const
{
    // Validate arrayant
    std::string error_message = qd_arrayant_is_valid_quick(*this);
    if (error_message.length() == 0 && i_element >= n_elements())
        error_message = "Element index out of bound.";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    qd_check_alias_list<dtype>({beamwidth_az, beamwidth_el, az_point_ang, el_point_ang},
                               {"beamwidth_az", "beamwidth_el", "az_point_ang", "el_point_ang"});

    const bool need_az = (beamwidth_az != nullptr) || (az_point_ang != nullptr);
    const bool need_el = (beamwidth_el != nullptr) || (el_point_ang != nullptr);
    if (!need_az && !need_el)
        return;

    const double rad2deg = 180.0 / arma::datum::pi;
    const double thres_lin = std::pow(10.0, -0.1 * double(threshold_dB));

    const arma::uword nEl = e_theta_re.n_rows;
    const arma::uword nAz = e_theta_re.n_cols;

    // Find (iEl, iAz) of the maximum on the coarse grid
    const dtype *p_tre = e_theta_re.slice_memptr(i_element);
    const dtype *p_tim = e_theta_im.slice_memptr(i_element);
    const dtype *p_pre = e_phi_re.slice_memptr(i_element);
    const dtype *p_pim = e_phi_im.slice_memptr(i_element);

    double p_max_coarse = -1.0;
    arma::uword i_max = 0;
    for (arma::uword i = 0; i < nEl * nAz; ++i)
    {
        double a = double(p_tre[i]), b = double(p_tim[i]);
        double c = double(p_pre[i]), d = double(p_pim[i]);
        double v = a * a + b * b + c * c + d * d;
        if (v > p_max_coarse)
        {
            p_max_coarse = v;
            i_max = i;
        }
    }
    const arma::uword iEl_max = i_max % nEl; // column-major: idx = iEl + iAz*nEl
    const arma::uword iAz_max = i_max / nEl;

    arma::uvec i_el_vec(1);
    i_el_vec(0) = i_element;

    // Width of the main lobe in samples of a cut: the samples above the threshold that are connected to the
    // maximum, searched outwards from the maximum (wrapping around for a circular cut). Returns n for a circular
    // cut that is above the threshold everywhere.
    auto lobe_width = [](const arma::Col<double> &P, arma::uword iM, double thres, bool circular) -> arma::uword
    {
        const arma::uword n = P.n_elem;
        arma::uword n_left = 0, n_right = 0;
        if (circular)
        {
            while (n_left + n_right + 1 < n && P(iM >= n_left + 1 ? iM - n_left - 1 : iM + n - n_left - 1) > thres)
                ++n_left;
            while (n_left + n_right + 1 < n && P((iM + n_right + 1) % n) > thres)
                ++n_right;
            if (n_left + n_right + 1 == n && P(iM) > thres)
                return n;
        }
        else
        {
            while (iM >= n_left + 1 && P(iM - n_left - 1) > thres)
                ++n_left;
            while (iM + n_right + 1 < n && P(iM + n_right + 1) > thres)
                ++n_right;
        }
        return n_left + n_right;
    };

    // Power of the interpolated cut and the index of its maximum
    auto cut_power = [&](const arma::Mat<dtype> &az_q, const arma::Mat<dtype> &el_q, arma::Col<double> &P) -> arma::uword
    {
        arma::Mat<dtype> V_re, V_im, H_re, H_im;
        quadriga_lib::arrayant_interpolate<dtype>(*this, az_q, el_q, V_re, V_im, H_re, H_im, &i_el_vec,
                                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
        P.set_size(az_q.n_elem);
        double P_max = 0.0;
        arma::uword iM = 0;
        for (arma::uword i = 0; i < az_q.n_elem; ++i)
        {
            double a = double(V_re(0, i)), b = double(V_im(0, i));
            double c = double(H_re(0, i)), d = double(H_im(0, i));
            double v = a * a + b * b + c * c + d * d;
            P(i) = v;
            if (v > P_max)
                P_max = v, iM = i;
        }
        return iM;
    };

    const double pi = arma::datum::pi, tau = 2.0 * pi, pi_half = 0.5 * pi, tol = 1.0e-4;

    // Azimuth pass: cut at the elevation of the maximum. When the grid covers the full circle (the gap between the
    // last and the first grid point is not larger than the largest grid step), the cut is circular.
    if (need_az)
    {
        const double az_first = double(azimuth_grid(0));
        const double az_last = double(azimuth_grid(nAz - 1));
        double step_max = 0.0;
        for (arma::uword i = 1; i < nAz; ++i)
            step_max = std::max(step_max, double(azimuth_grid(i) - azimuth_grid(i - 1)));
        const bool circular = nAz == 1 || az_first + tau - az_last <= step_max + tol;

        const arma::uword nAzi = circular ? 100 * nAz : 100 * nAz + 1;
        const double az_step = circular ? tau / double(nAzi) : (az_last - az_first) / double(100 * nAz);

        arma::Mat<dtype> az_q(1, nAzi), el_q(1, nAzi);
        for (arma::uword i = 0; i < nAzi; ++i)
        {
            double az = az_first + double(i) * az_step;
            az_q(0, i) = dtype(az > pi ? az - tau : az);
        }
        if (!circular)
            az_q(0, nAzi - 1) = dtype(az_last);
        el_q.fill(elevation_grid(iEl_max));

        arma::Col<double> Pi;
        const arma::uword iM = cut_power(az_q, el_q, Pi);
        const arma::uword width = lobe_width(Pi, iM, Pi(iM) * thres_lin, circular);

        if (beamwidth_az != nullptr)
            *beamwidth_az = dtype(width == nAzi ? 360.0 : double(width) * az_step * rad2deg);
        if (az_point_ang != nullptr)
            *az_point_ang = dtype(double(az_q(0, iM)) * rad2deg);
    }

    // Elevation pass: cut along the great circle through the maximum (azimuth of the maximum and the opposite
    // azimuth), so that a main lobe can extend across the poles. Without samples at a pole the cut ends there.
    // The result is at most 180 degree.
    if (need_el)
    {
        const double el_first = double(elevation_grid(0));
        const double el_last = double(elevation_grid(nEl - 1));
        const bool full_el = el_first <= -pi_half + tol && el_last >= pi_half - tol;

        const double az_front = double(azimuth_grid(iAz_max));
        const double az_back = az_front > 0.0 ? az_front - pi : az_front + pi;

        arma::uword nEli;
        double el_step;
        if (full_el) // Full great circle, parameter t in [el_first, el_first + 2 pi)
            nEli = 200 * nEl, el_step = tau / double(nEli);
        else
            nEli = 100 * nEl + 1, el_step = (el_last - el_first) / double(100 * nEl);

        arma::Mat<dtype> az_q(1, nEli), el_q(1, nEli);
        for (arma::uword i = 0; i < nEli; ++i)
        {
            double t = el_first + double(i) * el_step;
            if (!full_el || t <= pi_half) // Front half
                az_q(0, i) = dtype(az_front), el_q(0, i) = dtype(t);
            else if (t <= 3.0 * pi_half) // Back half, across the north pole
                az_q(0, i) = dtype(az_back), el_q(0, i) = dtype(pi - t);
            else // Across the south pole, back to the front half
                az_q(0, i) = dtype(az_front), el_q(0, i) = dtype(t - tau);
        }
        if (!full_el)
            el_q(0, nEli - 1) = dtype(el_last);

        arma::Col<double> Pi;
        const arma::uword iM = cut_power(az_q, el_q, Pi);
        const arma::uword width = lobe_width(Pi, iM, Pi(iM) * thres_lin, full_el);

        if (beamwidth_el != nullptr)
            *beamwidth_el = dtype(std::min(double(width) * el_step * rad2deg, 180.0));
        if (el_point_ang != nullptr)
            *el_point_ang = dtype(double(el_q(0, iM)) * rad2deg);
    }
}

/*!MD
# .calc_directivity_dBi
Calculate the directivity in dBi of a single array element

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Independent of the scale of the pattern; an element whose pattern is zero everywhere returns 0 dBi
- Ignores element coupling

## Declaration:
```
dtype quadriga_lib::arrayant<dtype>::calc_directivity_dBi(arma::uword i_element) const;
```

## Inputs:
- **`i_element`** — Element index, 0-based

## Returns:
- Directivity of the specified element in dBi

## See also:
- [[arrayant_combine_pattern]] (the per-port directivity is a typical follow-up)
MD!*/

template <typename dtype>
dtype quadriga_lib::arrayant<dtype>::calc_directivity_dBi(arma::uword i_element) const
{
    // Check if arrayant object is valid
    std::string error_message = qd_arrayant_is_valid_quick(*this);
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
        double y = i == nel - 1 ? pi_half : 0.5 * el.at(i) + 0.5 * el.at(i + 1);
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
    auto sum_power = [&](double s)
    {
        p_sum = 0.0, p_max = 0.0;
        for (arma::uword i = 0; i < naz * nel; ++i)
        {
            double a = s * double(p_theta_re[i]), b = s * double(p_theta_im[i]), c = s * double(p_phi_re[i]), d = s * double(p_phi_im[i]);
            double pow = a * a + b * b + c * c + d * d;
            p_max = pow > p_max ? pow : p_max;
            p_sum += pow * ptr[i];
        }
    };
    sum_power(1.0);
    if (std::isinf(p_max) || std::isinf(p_sum)) // Power above the double range
        sum_power(qd_inverse_max_field(p_theta_re, p_theta_im, p_phi_re, p_phi_im, naz * nel));

    // An all-zero element has no directivity; 0 dBi is returned
    double directivity = (p_max > 0.0 && p_sum > 0.0) ? 10.0 * std::log10(p_max / p_sum) : 0.0;
    return dtype(directivity);
}

// Creates a copy of the array antenna object
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::copy() const
{
    quadriga_lib::arrayant<dtype> ant(*this); // The copy of a valid arrayant is marked as validated
    ant.read_only = false;
    return ant;
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
# .set_size
Resize an arrayant object to new dimensions

- No-op if current dimensions already match
- After resize: element_pos is zeroed, coupling_re set to identity, coupling_im zeroed; all other field data is undefined

## Declaration:
```
void quadriga_lib::arrayant<dtype>::set_size(
    arma::uword n_elevation,
    arma::uword n_azimuth,
    arma::uword n_elements,
    arma::uword n_ports);
```

## Inputs:
- **`n_elevation`** — Number of elevation samples
- **`n_azimuth`** — Number of azimuth samples
- **`n_elements`** — Number of antenna elements
- **`n_ports`** — Number of ports (columns of coupling matrix)
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
Validate the integrity of an arrayant object

- Always checks the data: sizes of all members, ranges and order of the angle grids, sizes of element positions and
  coupling matrix; empty `element_pos` and coupling are allowed
- A passing check marks the object as validated: the library functions then skip the full check as long as no data
  member was reallocated or resized; values changed in place after the check are not seen by them
- The mark is updated atomically, so `is_valid` may be called while other threads read the object
- A copy is marked if its source passes the full check; a moved-to object takes over the mark of its source

## Declaration:
```
std::string quadriga_lib::arrayant<dtype>::is_valid() const;
```

## Returns:
- Empty string if valid; error message string if invalid

## See also:
- [[arrayant_is_valid_multi]] (multi-freq counterpart)
MD!*/

// Range and order of an angle grid in one pass; stops at the first value out of range
template <typename dtype>
static inline void qd_grid_range_order(const arma::Col<dtype> &grid, dtype max_abs, bool &in_range, bool &ascending)
{
    const dtype *p = grid.memptr();
    in_range = true, ascending = true;
    for (arma::uword i = 0; i < grid.n_elem; ++i)
    {
        if (!(p[i] >= -max_abs && p[i] <= max_abs))
        {
            in_range = false;
            return;
        }
        if (i != 0 && p[i] <= p[i - 1])
            ascending = false;
    }
}

// Full check of an arrayant without writing to it
template <typename dtype>
std::string qd_arrayant_check(const quadriga_lib::arrayant<dtype> &ant)
{
    const auto &e_theta_re = ant.e_theta_re, &e_theta_im = ant.e_theta_im, &e_phi_re = ant.e_phi_re, &e_phi_im = ant.e_phi_im;
    const auto &azimuth_grid = ant.azimuth_grid, &elevation_grid = ant.elevation_grid;
    const auto &element_pos = ant.element_pos, &coupling_re = ant.coupling_re, &coupling_im = ant.coupling_im;

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

    bool az_in_range, az_ascending, el_in_range, el_ascending;
    qd_grid_range_order(azimuth_grid, dtype(3.1415930), az_in_range, az_ascending);
    qd_grid_range_order(elevation_grid, dtype(1.5707965), el_in_range, el_ascending);

    if (!az_in_range)
        return std::string("Values of 'azimuth_grid' must be between -pi and pi (equivalent to -180 to 180 degree).");

    if (!el_in_range)
        return std::string("Values of 'elevation_grid' must be between -pi/2 and pi/2 (equivalent to -90 to 90 degree).");

    if (!az_ascending)
        return std::string("Values of 'azimuth_grid' must be sorted in strictly ascending order.");

    if (!el_ascending)
        return std::string("Values of 'elevation_grid' must be sorted in strictly ascending order.");

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

template std::string qd_arrayant_check(const quadriga_lib::arrayant<float> &);
template std::string qd_arrayant_check(const quadriga_lib::arrayant<double> &);

// ARRAYANT METHOD : Full check; a passing check marks the object as validated
template <typename dtype>
std::string quadriga_lib::arrayant<dtype>::is_valid() const
{
    std::string error_message = qd_arrayant_check(*this);
    if (error_message.empty())
        qd_arrayant_mark_valid(*this);
    return error_message;
}

// The source gets the full check (read only), so a mark that no longer fits the data is not passed on.
template <typename dtype>
quadriga_lib::arrayant<dtype>::arrayant(const arrayant &other)
{
    *this = other;
}

template <typename dtype>
quadriga_lib::arrayant<dtype> &quadriga_lib::arrayant<dtype>::operator=(const arrayant &other)
{
    if (this == &other)
        return *this;
    qd_arrayant_clear_mark(*this); // Cleared first: a member copy that throws leaves an unmarked object
    name = other.name;
    e_theta_re = other.e_theta_re, e_theta_im = other.e_theta_im, e_phi_re = other.e_phi_re, e_phi_im = other.e_phi_im;
    azimuth_grid = other.azimuth_grid, elevation_grid = other.elevation_grid;
    element_pos = other.element_pos, coupling_re = other.coupling_re, coupling_im = other.coupling_im;
    center_frequency = other.center_frequency, read_only = other.read_only;
    if (qd_arrayant_check(other).empty())
        qd_arrayant_mark_valid(*this);
    return *this;
}

// Move: Armadillo takes over heap memory but copies small members (up to 16 values are stored inside the object),
// so data pointers can change. The target of a marked source is marked with its own pointers (no check, the target
// takes over the data of the source); the source loses its mark.
template <typename dtype>
quadriga_lib::arrayant<dtype>::arrayant(arrayant &&other)
{
    *this = std::move(other);
}

template <typename dtype>
quadriga_lib::arrayant<dtype> &quadriga_lib::arrayant<dtype>::operator=(arrayant &&other)
{
    if (this == &other)
        return *this;
    const bool marked = qd_arrayant_cache_matches(other);
    qd_arrayant_clear_mark(*this), qd_arrayant_clear_mark(other);
    name = std::move(other.name);
    e_theta_re = std::move(other.e_theta_re), e_theta_im = std::move(other.e_theta_im);
    e_phi_re = std::move(other.e_phi_re), e_phi_im = std::move(other.e_phi_im);
    azimuth_grid = std::move(other.azimuth_grid), elevation_grid = std::move(other.elevation_grid);
    element_pos = std::move(other.element_pos), coupling_re = std::move(other.coupling_re);
    coupling_im = std::move(other.coupling_im);
    center_frequency = other.center_frequency, read_only = other.read_only;
    if (marked)
        qd_arrayant_mark_valid(*this);
    return *this;
}

// The validity cache must not need a lock (concurrent is_valid and quick checks; no libatomic)
static_assert(std::atomic<const float *>::is_always_lock_free && std::atomic<const double *>::is_always_lock_free,
              "Atomic pointers are not lock-free on this platform.");

// Instantiate templates
template class quadriga_lib::arrayant<float>;
template class quadriga_lib::arrayant<double>;