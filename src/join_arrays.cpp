// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Block-diagonal port weights [W1 0; 0 W2] of two arrays with n_el1 and n_el2 elements. An empty port_weights_re is an
// identity block (one port per element); an empty port_weights_im is zero. Outputs are set to
// [n_el1 + n_el2, n_ports1 + n_ports2]; they must not alias the inputs.
template <typename dtype>
static void qd_port_weights_block_diag(const arma::Mat<dtype> &port_weights_re1, const arma::Mat<dtype> &port_weights_im1,
                                            arma::uword n_el1,
                                            const arma::Mat<dtype> &port_weights_re2, const arma::Mat<dtype> &port_weights_im2,
                                            arma::uword n_el2,
                                            arma::Mat<dtype> &port_weights_re, arma::Mat<dtype> &port_weights_im)
{
    const arma::uword n_ports1 = port_weights_re1.is_empty() ? n_el1 : port_weights_re1.n_cols;
    const arma::uword n_ports2 = port_weights_re2.is_empty() ? n_el2 : port_weights_re2.n_cols;
    port_weights_re.zeros(n_el1 + n_el2, n_ports1 + n_ports2);
    port_weights_im.zeros(n_el1 + n_el2, n_ports1 + n_ports2);

    if (n_el1 != 0 && n_ports1 != 0)
    {
        if (port_weights_re1.is_empty())
            port_weights_re.submat(0, 0, n_el1 - 1, n_ports1 - 1).eye();
        else
            port_weights_re.submat(0, 0, n_el1 - 1, n_ports1 - 1) = port_weights_re1;
        if (!port_weights_im1.is_empty())
            port_weights_im.submat(0, 0, n_el1 - 1, n_ports1 - 1) = port_weights_im1;
    }
    if (n_el2 != 0 && n_ports2 != 0)
    {
        const arma::uword r1 = n_el1 + n_el2 - 1, c1 = n_ports1 + n_ports2 - 1;
        if (port_weights_re2.is_empty())
            port_weights_re.submat(n_el1, n_ports1, r1, c1).eye();
        else
            port_weights_re.submat(n_el1, n_ports1, r1, c1) = port_weights_re2;
        if (!port_weights_im2.is_empty())
            port_weights_im.submat(n_el1, n_ports1, r1, c1) = port_weights_im2;
    }
}

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

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

## Declarations:
```
quadriga_lib::arrayant<dtype> quadriga_lib::join_arrays(
    const arrayant<dtype> &ant_1,
    const arrayant<dtype> &ant_2);

std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::join_arrays(
    const std::vector<arrayant<dtype>> &arrayant_vec1,
    const std::vector<arrayant<dtype>> &arrayant_vec2);
```

## Inputs:
- **`ant_1`** / **`arrayant_vec1`** — First array; its elements come first in the output; vector form: must pass
  [[is_valid]]
- **`ant_2`** / **`arrayant_vec2`** — Second array with the same sampling grids; vector form: must pass [[is_valid]],
  with as many entries as the first and matching `center_freq` values at each index

## Returns:
- New `arrayant` with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports; vector form: one such entry per
  frequency

## See also:
- [[set_element_pos]] (positions the drivers of a multi-frequency array)
- [[copy_element]] (duplicates an element within one array)
- [[generate_speaker]] (creates the driver models to join)
- [[qdant_write]] (stores the joined model)
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::join_arrays(const arrayant<dtype> &ant_1, const arrayant<dtype> &ant_2)
{
    std::string error_message = qd_arrayant_is_valid_quick(ant_1);
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'ant_1' is not valid: " + error_message);
    error_message = qd_arrayant_is_valid_quick(ant_2);
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'ant_2' is not valid: " + error_message);

    // Check if grids are identical
    if (!arma::approx_equal(ant_1.azimuth_grid, ant_2.azimuth_grid, "absdiff", (dtype)1.0e-6))
        throw std::invalid_argument("Inputs 'ant_1' and 'ant_2' must have the same azimuth grid.");
    if (!arma::approx_equal(ant_1.elevation_grid, ant_2.elevation_grid, "absdiff", (dtype)1.0e-6))
        throw std::invalid_argument("Inputs 'ant_1' and 'ant_2' must have the same elevation grid.");

    arma::uword n_elevation = ant_1.n_elevation();
    arma::uword n_azimuth = ant_1.n_azimuth();
    arma::uword n_elements_1 = ant_1.n_elements();
    arma::uword n_elements_2 = ant_2.n_elements();
    arma::uword n_ports_1 = ant_1.n_ports();
    arma::uword n_ports_2 = ant_2.n_ports();
    arma::uword n_ang = n_azimuth * n_elevation;

    quadriga_lib::arrayant<dtype> output;
    output.set_size(n_elevation, n_azimuth, n_elements_1 + n_elements_2, n_ports_1 + n_ports_2);

    // Copy data from first antenna
    std::memcpy(output.azimuth_grid.memptr(), ant_1.azimuth_grid.memptr(), n_azimuth * sizeof(dtype));
    std::memcpy(output.elevation_grid.memptr(), ant_1.elevation_grid.memptr(), n_elevation * sizeof(dtype));
    std::memcpy(output.e_theta_re.slice_memptr(0), ant_1.e_theta_re.memptr(), n_ang * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_theta_im.slice_memptr(0), ant_1.e_theta_im.memptr(), n_ang * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_phi_re.slice_memptr(0), ant_1.e_phi_re.memptr(), n_ang * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_phi_im.slice_memptr(0), ant_1.e_phi_im.memptr(), n_ang * n_elements_1 * sizeof(dtype));
    if (!ant_1.element_pos.empty())
        std::memcpy(output.element_pos.colptr(0), ant_1.element_pos.memptr(), 3 * n_elements_1 * sizeof(dtype));

    // Inherit scalar/string metadata from ant_1
    output.center_freq = ant_1.center_freq;
    output.name = ant_1.name;

    // Copy data from second antenna
    std::memcpy(output.e_theta_re.slice_memptr(n_elements_1), ant_2.e_theta_re.memptr(), n_ang * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_theta_im.slice_memptr(n_elements_1), ant_2.e_theta_im.memptr(), n_ang * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_phi_re.slice_memptr(n_elements_1), ant_2.e_phi_re.memptr(), n_ang * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_phi_im.slice_memptr(n_elements_1), ant_2.e_phi_im.memptr(), n_ang * n_elements_2 * sizeof(dtype));
    if (!ant_2.element_pos.empty())
        std::memcpy(output.element_pos.colptr(n_elements_1), ant_2.element_pos.memptr(), 3 * n_elements_2 * sizeof(dtype));

    // Block-diagonal coupling (empty port_weights_re: identity block, empty port_weights_im: zeros)
    qd_port_weights_block_diag(ant_1.port_weights_re, ant_1.port_weights_im, n_elements_1,
                                    ant_2.port_weights_re, ant_2.port_weights_im, n_elements_2,
                                    output.port_weights_re, output.port_weights_im);

    return output;
}

template quadriga_lib::arrayant<float> quadriga_lib::join_arrays(const arrayant<float> &, const arrayant<float> &);
template quadriga_lib::arrayant<double> quadriga_lib::join_arrays(const arrayant<double> &, const arrayant<double> &);

template <typename dtype>
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::join_arrays(
    const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec1,
    const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec2)
{
    // --- Validate individual inputs ---
    std::string err = qd_arrayant_vec_check(arrayant_vec1);
    if (!err.empty())
        throw std::invalid_argument("Input 'arrayant_vec1' is not valid: " + err);

    err = qd_arrayant_vec_check(arrayant_vec2);
    if (!err.empty())
        throw std::invalid_argument("Input 'arrayant_vec2' is not valid: " + err);

    // --- Check frequency count ---
    arma::uword n_freq = (arma::uword)arrayant_vec1.size();
    if ((arma::uword)arrayant_vec2.size() != n_freq)
        throw std::invalid_argument("Inputs 'arrayant_vec1' and 'arrayant_vec2' have different numbers of frequency entries (" +
                                    std::to_string(arrayant_vec1.size()) + " vs " +
                                    std::to_string(arrayant_vec2.size()) + ").");

    // --- Check angular grid compatibility (compare against first entry of vec1) ---
    arma::uword n_el = arrayant_vec1[0].n_elevation();
    arma::uword n_az = arrayant_vec1[0].n_azimuth();

    if (arrayant_vec2[0].n_elevation() != n_el)
        throw std::invalid_argument("Inputs 'arrayant_vec1' and 'arrayant_vec2' must have the same elevation grid size (" +
                                    std::to_string(n_el) + " vs " +
                                    std::to_string(arrayant_vec2[0].n_elevation()) + ").");

    if (arrayant_vec2[0].n_azimuth() != n_az)
        throw std::invalid_argument("Inputs 'arrayant_vec1' and 'arrayant_vec2' must have the same azimuth grid size (" +
                                    std::to_string(n_az) + " vs " +
                                    std::to_string(arrayant_vec2[0].n_azimuth()) + ").");

    const dtype *az_ref = arrayant_vec1[0].azimuth_grid.memptr();
    const dtype *az_cur = arrayant_vec2[0].azimuth_grid.memptr();
    for (arma::uword k = 0; k < n_az; ++k)
        if (az_ref[k] != az_cur[k])
            throw std::invalid_argument("Inputs 'arrayant_vec1' and 'arrayant_vec2' must have the same azimuth grid.");

    const dtype *el_ref = arrayant_vec1[0].elevation_grid.memptr();
    const dtype *el_cur = arrayant_vec2[0].elevation_grid.memptr();
    for (arma::uword k = 0; k < n_el; ++k)
        if (el_ref[k] != el_cur[k])
            throw std::invalid_argument("Inputs 'arrayant_vec1' and 'arrayant_vec2' must have the same elevation grid.");

    // --- Reference dimensions ---
    arma::uword n_elem1 = arrayant_vec1[0].n_elements();
    arma::uword n_elem2 = arrayant_vec2[0].n_elements();

    // --- Build output ---
    std::vector<quadriga_lib::arrayant<dtype>> output(n_freq);

    for (arma::uword i = 0; i < n_freq; ++i)
    {
        const quadriga_lib::arrayant<dtype> &a1 = arrayant_vec1[i];
        const quadriga_lib::arrayant<dtype> &a2 = arrayant_vec2[i];

        // Check center frequency match
        if (std::abs((double)a1.center_freq - (double)a2.center_freq) > 1.0e-6 * std::abs((double)a1.center_freq))
            throw std::invalid_argument("Entry " + std::to_string(i) + ": Center frequency mismatch (" + std::to_string((double)a1.center_freq) +
                                        " vs " + std::to_string((double)a2.center_freq) + ").");

        quadriga_lib::arrayant<dtype> &out = output[i];

        // Copy grids and metadata from vec1
        out.name = a1.name;
        out.azimuth_grid = a1.azimuth_grid;
        out.elevation_grid = a1.elevation_grid;
        out.center_freq = a1.center_freq;

        // Concatenate patterns along the element (slice) dimension
        out.e_theta_re = arma::join_slices(a1.e_theta_re, a2.e_theta_re);
        out.e_theta_im = arma::join_slices(a1.e_theta_im, a2.e_theta_im);
        out.e_phi_re = arma::join_slices(a1.e_phi_re, a2.e_phi_re);
        out.e_phi_im = arma::join_slices(a1.e_phi_im, a2.e_phi_im);

        // Concatenate element positions
        arma::Mat<dtype> pos1 = a1.element_pos.is_empty() ? arma::Mat<dtype>(3, n_elem1, arma::fill::zeros) : a1.element_pos;
        arma::Mat<dtype> pos2 = a2.element_pos.is_empty() ? arma::Mat<dtype>(3, n_elem2, arma::fill::zeros) : a2.element_pos;
        out.element_pos = arma::join_horiz(pos1, pos2);

        // Block-diagonal coupling matrix assembly
        // [ C1  0  ]
        // [ 0   C2 ]
        qd_port_weights_block_diag(a1.port_weights_re, a1.port_weights_im, n_elem1,
                                        a2.port_weights_re, a2.port_weights_im, n_elem2,
                                        out.port_weights_re, out.port_weights_im);

        // The check also marks the entry as validated
        err = out.is_valid();
        if (!err.empty())
            throw std::invalid_argument("Entry " + std::to_string(i) + ": Output validation failed: " + err);
    }

    return output;
}

template std::vector<quadriga_lib::arrayant<float>> quadriga_lib::join_arrays(
    const std::vector<quadriga_lib::arrayant<float>> &, const std::vector<quadriga_lib::arrayant<float>> &);

template std::vector<quadriga_lib::arrayant<double>> quadriga_lib::join_arrays(
    const std::vector<quadriga_lib::arrayant<double>> &, const std::vector<quadriga_lib::arrayant<double>> &);
