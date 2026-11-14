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
| Function      | Description                           |
| ------------- | ------------------------------------- |
| .[[set_size]] | Resize the arrayant to new dimensions |
| .[[is_valid]] | Validate arrayant integrity           |

Computations and file I/O on an arrayant are free functions, see section "Array antenna functions"
(e.g. [[arrayant_calc_beamwidth]], [[arrayant_calc_directivity]], [[arrayant_concat]], [[arrayant_combine_pattern]],
[[arrayant_copy_element]], [[arrayant_rotate_pattern]], [[qdant_write]]).
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