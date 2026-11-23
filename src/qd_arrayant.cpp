// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <stdexcept>
#include <string>
#include <vector>

#include "quadriga_arrayant.hpp"
#include "quadriga_tools.hpp"
#include "quadriga_math.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!SECTION_DESC
The functions of this section create, inspect, transform and store antenna and acoustic arrays. They share one data
structure, [[arrayant]]: complex polarimetric patterns of the elements on an azimuth and elevation grid, the element
positions and the port weights. The channel functions take it as transmit and receive antenna.<br><br>

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
- **Port weights**: the fixed complex feed network $w_{i,p}$ from element $i$ to port $p$, `[n_elements, n_ports]`. It
  is not the mutual coupling between elements. Empty means identity (one port per element)
- **Element position**: local Cartesian coordinates in m, `[3, n_elements]`; empty means all at the origin
- **Multi-frequency array**: a `std::vector` of arrayants, one entry per `center_freq`, with equal grids, element
  positions and port weight sizes ([[is_valid]]). [[interpolate_pattern]] and [[apply_port_weights]] interpolate over
  frequency between the entries

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
| Transform | [[apply_port_weights]], [[rotate_pattern]], [[trim_pattern]] |
| Edit and combine | [[join_arrays]], [[copy_element]], [[set_element_pos]] |
| Check, store, show | [[is_valid]], [[qdant_write]], [[export_pattern_obj]] |

Arrays are stored in the [QDANT file format](formats.html#qdant).
SECTION_DESC!*/

/*!MD
# arrayant<++>
Class for storing and manipulating an array of antenna or acoustic elements

- All functions give the same results for an empty `element_pos` and empty port weights as for zeros and identity,
  except the IRS array of [[get_channels_irs]], whose port weights hold the IRS weights and must be given
- In a multi-frequency vector all entries use the same form, either empty or given (see [[is_valid]])

## Attributes:<br>
| Attribute                         | Size                                   | Description                                                    |
| --------------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| `arma::Cube<dtype> e_theta_re`    | `[n_elevation, n_azimuth, n_elements]` | e-theta (vertical) field, real part                            |
| `arma::Cube<dtype> e_theta_im`    | `[n_elevation, n_azimuth, n_elements]` | e-theta (vertical) field, imaginary part                       |
| `arma::Cube<dtype> e_phi_re`      | `[n_elevation, n_azimuth, n_elements]` | e-phi (horizontal) field, real part                            |
| `arma::Cube<dtype> e_phi_im`      | `[n_elevation, n_azimuth, n_elements]` | e-phi (horizontal) field, imaginary part                       |
| `arma::Col<dtype> azimuth_grid`   | `[n_azimuth]`                          | Azimuth angles in rad, in [-pi, pi], strictly increasing       |
| `arma::Col<dtype> elevation_grid` | `[n_elevation]`                        | Elevation angles in rad, in [-pi/2, pi/2], strictly increasing |
| `arma::Mat<dtype> element_pos`    | `[3, n_elements]` or empty             | Element positions relative to the phase center, in m           |
| `arma::Mat<dtype> port_weights_re`    | `[n_elements, n_ports]` or empty       | Port weights, real part                                     |
| `arma::Mat<dtype> port_weights_im`    | `[n_elements, n_ports]` or empty       | Port weights, imaginary part                                |
| `dtype center_freq`          | scalar                                 | Center frequency in Hz; default 299792458                      |
| `std::string name`                | string                                 | Name of the antenna array object                               |

## Member functions:<br>
| Function                                                  | Description                                                                                                                                                         |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.n_elevation()`                                          | Number of elevation angles                                                                                                                                          |
| `.n_azimuth()`                                            | Number of azimuth angles                                                                                                                                            |
| `.n_elements()`                                           | Number of antenna elements                                                                                                                                          |
| `.n_ports()`                                              | Number of ports (columns of port weights)                                                                                                                        |
| `.reset()`                                                | Empties all data; `center_freq` returns to its default, `name` is kept                                                                                              |
| `.set_size(n_elevation, n_azimuth, n_elements, n_ports)` | Resizes all members (`arma::uword` sizes); no-op if they match; after a resize `element_pos` is zero, `port_weights_re` identity, `port_weights_im` zero, field data undefined |
| `.is_valid()`                                             | `std::string`: empty if the object is valid, else an error message                                                                                                  |

## Notes:
- **Value range**: field values and port weights must be finite with magnitudes below about 1e18; larger values overflow
  in the interpolation and in the channel functions and give NaN results
- **Validation**: `is_valid()` always checks the data: sizes of all members, ranges and order of the angle grids, sizes
  of element positions and port weights. A passing check marks the object as validated: the library functions then skip
  the full check as long as no data member was reallocated or resized; values changed in place after the check are not
  seen by them
- **Threads and copies**: the mark is updated atomically, so `is_valid()` may be called while other threads read the
  object; a copy is marked if its source passes the full check, a moved-to object takes over the mark of its source

## See also:
- [[is_valid]] (checks a multi-frequency vector)
- [[generate_antenna]], [[generate_speaker]] (create arrays)
- [[qdant_read]], [[qdant_write]] (load and store arrays)
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
    if (port_weights_re.empty() && port_weights_im.empty())
        return e_theta_re.n_slices;
    else if (port_weights_re.empty())
        return port_weights_im.n_cols;
    else
        return port_weights_re.n_cols;
}

// Empties all members
template <typename dtype>
void quadriga_lib::arrayant<dtype>::reset()
{
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

    if (port_weights_re.n_elem != 0)
        port_weights_re.reset();

    if (port_weights_im.n_elem != 0)
        port_weights_im.reset();

    center_freq = dtype(qd_speed_of_light);
}

template <typename dtype>
void quadriga_lib::arrayant<dtype>::set_size(arma::uword n_elevation, arma::uword n_azimuth,
                                             arma::uword n_elements, arma::uword n_ports)
{
    azimuth_grid.set_size(n_azimuth);
    elevation_grid.set_size(n_elevation);
    e_theta_re.set_size(n_elevation, n_azimuth, n_elements);
    e_theta_im.set_size(n_elevation, n_azimuth, n_elements);
    e_phi_re.set_size(n_elevation, n_azimuth, n_elements);
    e_phi_im.set_size(n_elevation, n_azimuth, n_elements);

    // Zeroed / identity only when the size changes

    if (element_pos.n_rows != 3 || element_pos.n_cols != n_elements)
        element_pos.zeros(3, n_elements);

    if (port_weights_re.n_rows != n_elements || port_weights_re.n_cols != n_ports)
        port_weights_re.eye(n_elements, n_ports);

    if (port_weights_im.n_rows != n_elements || port_weights_im.n_cols != n_ports)
        port_weights_im.zeros(n_elements, n_ports);
}

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
    const auto &element_pos = ant.element_pos, &port_weights_re = ant.port_weights_re, &port_weights_im = ant.port_weights_im;

    if (e_theta_re.n_elem == 0ULL || e_theta_im.n_elem == 0ULL || e_phi_re.n_elem == 0ULL || e_phi_im.n_elem == 0ULL ||
        azimuth_grid.n_elem == 0ULL || elevation_grid.n_elem == 0ULL)
        return std::string("Missing data for any of: 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im', 'azimuth_grid', 'elevation_grid'.");

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
        return std::string("Size of 'element_pos' must be either empty or match [3, n_elements].");

    if (!port_weights_re.empty() && port_weights_re.n_rows != n_elements)
        return std::string("'port_weights_re' must have one row per element.");

    if (port_weights_re.empty() && !port_weights_im.empty())
        return std::string("Imaginary part of port weights (phase component) defined without real part (absolute component).");

    if (!port_weights_im.empty() && (port_weights_im.n_rows != n_elements || port_weights_im.n_cols != port_weights_re.n_cols))
        return std::string("'port_weights_im' must be empty or its size must match 'port_weights_re'.");

    return std::string("");
}

template std::string qd_arrayant_check(const quadriga_lib::arrayant<float> &);
template std::string qd_arrayant_check(const quadriga_lib::arrayant<double> &);

// A passing check marks the object as validated
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
    element_pos = other.element_pos, port_weights_re = other.port_weights_re, port_weights_im = other.port_weights_im;
    center_freq = other.center_freq;
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
    element_pos = std::move(other.element_pos), port_weights_re = std::move(other.port_weights_re);
    port_weights_im = std::move(other.port_weights_im);
    center_freq = other.center_freq;
    if (marked)
        qd_arrayant_mark_valid(*this);
    return *this;
}

// The validity cache must not need a lock (concurrent is_valid and quick checks; no libatomic)
static_assert(std::atomic<const float *>::is_always_lock_free && std::atomic<const double *>::is_always_lock_free,
              "Atomic pointers are not lock-free on this platform.");

template class quadriga_lib::arrayant<float>;
template class quadriga_lib::arrayant<double>;

/*!MD
# is_valid
Validate a vector of arrayant objects for multi-frequency consistency

- Checks that the entries can be used together as one multi-frequency array and stops at the first error
- Each entry gets the full check of its `is_valid` member, which marks a passing entry as validated. It is then compared
  with the first entry: size and values of the angle grids, number of elements, element positions, size of the port
  weights
- All entries must use the same form for `element_pos`, `port_weights_re` and `port_weights_im`: all empty or all given;
  a mix of empty members and explicit zeros or identity is rejected
- Pattern data, `center_freq` and port weight values are not compared, since they vary with frequency

## Declaration:
```
std::string quadriga_lib::is_valid(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

## Input:
- **`arrayant_vec`** — Vector of arrayant objects to validate; an empty vector is reported as invalid

## Returns:
- Empty string if valid; otherwise a message that names the failing entry and property, such as
  `"Entry 3: Azimuth grid values do not match the first entry."`

## See also:
- [[arrayant]] (its `is_valid()` member checks a single object)
- [[set_element_pos]] (keeps the positions of all entries equal)
- [[generate_speaker]] (typical source of multi-frequency arrayant vectors)
MD!*/

// Checks of is_valid; per entry the public check (marks), the read-only full check or the read-only
// quick check
enum class qd_vec_check { mark, full, quick };
template <typename dtype>
static std::string qd_arrayant_vec_validate(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec, qd_vec_check mode)
{
    auto check = [mode](const quadriga_lib::arrayant<dtype> &ant)
    {
        return mode == qd_vec_check::mark ? ant.is_valid()
                                            : (mode == qd_vec_check::full ? qd_arrayant_check(ant) : qd_arrayant_is_valid_quick(ant));
    };

    if (arrayant_vec.empty())
        return std::string("The vector is empty.");

    // Validate the reference entry
    std::string err = check(arrayant_vec[0]);
    if (!err.empty())
        return std::string("Entry 0: ") + err;

    // Reference dimensions and pointers from the first entry
    arma::uword n_el_ref = arrayant_vec[0].n_elevation();
    arma::uword n_az_ref = arrayant_vec[0].n_azimuth();
    arma::uword n_elements_ref = arrayant_vec[0].n_elements();
    arma::uword n_cpl_rows_ref = arrayant_vec[0].port_weights_re.n_rows;
    arma::uword n_cpl_cols_ref = arrayant_vec[0].port_weights_re.n_cols;
    bool has_element_pos_ref = !arrayant_vec[0].element_pos.empty();
    bool has_port_weights_im_ref = !arrayant_vec[0].port_weights_im.empty();

    for (size_t i = 1; i < arrayant_vec.size(); ++i)
    {
        std::string idx = std::to_string(i);

        // Validate individual entry
        err = check(arrayant_vec[i]);
        if (!err.empty())
            return std::string("Entry ") + idx + ": " + err;

        // Check angular grid sizes
        if (arrayant_vec[i].n_elevation() != n_el_ref)
            return std::string("Entry ") + idx + ": Number of elevation angles (" +
                   std::to_string(arrayant_vec[i].n_elevation()) + ") does not match the first entry (" +
                   std::to_string(n_el_ref) + ").";

        if (arrayant_vec[i].n_azimuth() != n_az_ref)
            return std::string("Entry ") + idx + ": Number of azimuth angles (" +
                   std::to_string(arrayant_vec[i].n_azimuth()) + ") does not match the first entry (" +
                   std::to_string(n_az_ref) + ").";

        // Check angular grid values
        const dtype *az_ref = arrayant_vec[0].azimuth_grid.memptr();
        const dtype *az_cur = arrayant_vec[i].azimuth_grid.memptr();
        for (arma::uword k = 0; k < n_az_ref; ++k)
            if (az_ref[k] != az_cur[k])
                return std::string("Entry ") + idx + ": Azimuth grid values do not match the first entry.";

        const dtype *el_ref = arrayant_vec[0].elevation_grid.memptr();
        const dtype *el_cur = arrayant_vec[i].elevation_grid.memptr();
        for (arma::uword k = 0; k < n_el_ref; ++k)
            if (el_ref[k] != el_cur[k])
                return std::string("Entry ") + idx + ": Elevation grid values do not match the first entry.";

        // Check number of elements
        if (arrayant_vec[i].n_elements() != n_elements_ref)
            return std::string("Entry ") + idx + ": Number of elements (" +
                   std::to_string(arrayant_vec[i].n_elements()) + ") does not match the first entry (" +
                   std::to_string(n_elements_ref) + ").";

        // Check element positions
        bool has_element_pos_cur = !arrayant_vec[i].element_pos.empty();
        if (has_element_pos_ref != has_element_pos_cur)
            return std::string("Entry ") + idx + ": Presence of element_pos does not match the first entry.";

        if (has_element_pos_ref && has_element_pos_cur)
        {
            if (arrayant_vec[i].element_pos.n_rows != arrayant_vec[0].element_pos.n_rows ||
                arrayant_vec[i].element_pos.n_cols != arrayant_vec[0].element_pos.n_cols)
                return std::string("Entry ") + idx + ": Size of element_pos does not match the first entry.";

            const dtype *pos_ref = arrayant_vec[0].element_pos.memptr();
            const dtype *pos_cur = arrayant_vec[i].element_pos.memptr();
            arma::uword n_pos = arrayant_vec[0].element_pos.n_elem;
            for (arma::uword k = 0; k < n_pos; ++k)
                if (pos_ref[k] != pos_cur[k])
                    return std::string("Entry ") + idx + ": Element positions do not match the first entry.";
        }

        // Check port weights shape
        if (arrayant_vec[i].port_weights_re.n_rows != n_cpl_rows_ref ||
            arrayant_vec[i].port_weights_re.n_cols != n_cpl_cols_ref)
            return std::string("Entry ") + idx + ": Port weights size (" +
                   std::to_string(arrayant_vec[i].port_weights_re.n_rows) + "x" +
                   std::to_string(arrayant_vec[i].port_weights_re.n_cols) + ") does not match the first entry (" +
                   std::to_string(n_cpl_rows_ref) + "x" + std::to_string(n_cpl_cols_ref) + ").";

        bool has_port_weights_im_cur = !arrayant_vec[i].port_weights_im.empty();
        if (has_port_weights_im_ref != has_port_weights_im_cur)
            return std::string("Entry ") + idx + ": Presence of port_weights_im does not match the first entry.";

        if (has_port_weights_im_ref && has_port_weights_im_cur)
            if (arrayant_vec[i].port_weights_im.n_rows != arrayant_vec[0].port_weights_im.n_rows ||
                arrayant_vec[i].port_weights_im.n_cols != arrayant_vec[0].port_weights_im.n_cols)
                return std::string("Entry ") + idx + ": Size of port_weights_im does not match the first entry.";
    }

    return std::string("");
}

template <typename dtype>
std::string quadriga_lib::is_valid(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec)
{
    return qd_arrayant_vec_validate(arrayant_vec, qd_vec_check::mark);
}

template <typename dtype>
std::string qd_arrayant_vec_is_valid_quick(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec)
{
    return qd_arrayant_vec_validate(arrayant_vec, qd_vec_check::quick);
}

template <typename dtype>
std::string qd_arrayant_vec_check(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec)
{
    return qd_arrayant_vec_validate(arrayant_vec, qd_vec_check::full);
}

template std::string quadriga_lib::is_valid(const std::vector<quadriga_lib::arrayant<float>> &);
template std::string quadriga_lib::is_valid(const std::vector<quadriga_lib::arrayant<double>> &);
template std::string qd_arrayant_vec_is_valid_quick(const std::vector<quadriga_lib::arrayant<float>> &);
template std::string qd_arrayant_vec_is_valid_quick(const std::vector<quadriga_lib::arrayant<double>> &);
template std::string qd_arrayant_vec_check(const std::vector<quadriga_lib::arrayant<float>> &);
template std::string qd_arrayant_vec_check(const std::vector<quadriga_lib::arrayant<double>> &);