// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# copy_element
Copy a single element of an antenna array to one or more destination slots

- Copies the pattern and the position of one element in place, e.g. to build an array from a single element; the port
  weights of existing elements do not change
- A destination beyond the last element enlarges the array. Each added element gets its own new port with weight 1;
  added elements that are not a destination have a zero pattern and position

## Declarations:
```
void quadriga_lib::copy_element(
    arrayant<dtype> &ant,
    arma::uword i_source,
    const arma::uvec &i_target);

void quadriga_lib::copy_element(
    std::vector<arrayant<dtype>> &arrayant_vec,
    arma::uword i_source,
    const arma::uvec &i_target);
```

## Inputs:
- **`ant`** — Valid arrayant; updated in place
- **`arrayant_vec`** — Vector form: non-empty vector of valid arrayant objects; the same copy is made in every entry;
  updated in place
- **`i_source`** — Index of the element to copy, 0-based; must be below `n_elements`
- **`i_target`** — Destination indices, 0-based, not empty; a single index is passed as `{i}`; the largest `arma::uword`
  (e.g. `-1` cast to unsigned) throws

## Example:
```
auto ant = quadriga_lib::generate_antenna<double>("dipole");
quadriga_lib::copy_element(ant, 0, {1, 2, 3}); // 4 identical elements
ant.element_pos.row(1) = arma::rowvec({-0.75, -0.25, 0.25, 0.75});
```

## Notes:
- **Empty members**: when the array grows, an empty `element_pos` and empty port weights are first written out as zeros
  and identity, so that all elements can be edited afterward; otherwise they stay empty

## See also:
- [[set_element_pos]] (sets the positions of the copies in all entries of a vector)
- [[join_arrays]] (joins two arrays)
MD!*/

template <typename dtype>
void quadriga_lib::copy_element(arrayant<dtype> &ant, arma::uword i_source, const arma::uvec &i_target)
{
    if (i_target.n_elem == 0)
        throw std::invalid_argument("Input 'i_target' cannot be empty.");

    // The largest index cannot be reached by resizing: index + 1 would wrap to 0
    if (i_target.max() == std::numeric_limits<arma::uword>::max())
        throw std::invalid_argument("Input 'i_target' is out of range.");

    std::string error_message = qd_arrayant_check(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);

    arma::uword n_el = ant.e_theta_re.n_rows;
    arma::uword n_az = ant.e_theta_re.n_cols;
    arma::uword n_ang = n_el * n_az;
    arma::uword n_elements = ant.e_theta_re.n_slices;
    arma::uword n_element_max = i_target.max() + 1;

    if (i_source >= n_elements)
        throw std::invalid_argument("Input 'i_source' is out of range.");

    // Enlarge the arrayant; empty positions and port weights are written out first (zeros, identity), so that the new
    // elements can be edited; resize keeps the values and sets new ones to zero
    if (n_element_max > n_elements)
    {
        if (ant.element_pos.empty())
            ant.element_pos.zeros(3, n_elements);
        if (ant.port_weights_re.empty())
            ant.port_weights_re.eye(n_elements, n_elements);
        if (ant.port_weights_im.empty())
            ant.port_weights_im.zeros(arma::size(ant.port_weights_re));

        arma::uword added_elements = n_element_max - n_elements;
        arma::uword n_ports = ant.port_weights_re.n_cols;
        ant.e_theta_re.resize(n_el, n_az, n_element_max);
        ant.e_theta_im.resize(n_el, n_az, n_element_max);
        ant.e_phi_re.resize(n_el, n_az, n_element_max);
        ant.e_phi_im.resize(n_el, n_az, n_element_max);
        ant.element_pos.resize(3, n_element_max);
        ant.port_weights_re.resize(n_element_max, n_ports + added_elements);
        ant.port_weights_im.resize(n_element_max, n_ports + added_elements);
        for (arma::uword i = 0; i < added_elements; ++i)
            ant.port_weights_re.at(n_elements + i, n_ports + i) = (dtype)1.0;
    }

    // Copy data from source to destination
    for (auto dest = i_target.begin(); dest != i_target.end(); ++dest)
        if (i_source != *dest)
        {
            std::memcpy(ant.e_theta_re.slice_memptr(*dest), ant.e_theta_re.slice_memptr(i_source), n_ang * sizeof(dtype));
            std::memcpy(ant.e_theta_im.slice_memptr(*dest), ant.e_theta_im.slice_memptr(i_source), n_ang * sizeof(dtype));
            std::memcpy(ant.e_phi_re.slice_memptr(*dest), ant.e_phi_re.slice_memptr(i_source), n_ang * sizeof(dtype));
            std::memcpy(ant.e_phi_im.slice_memptr(*dest), ant.e_phi_im.slice_memptr(i_source), n_ang * sizeof(dtype));
            if (!ant.element_pos.empty()) // Empty only when the array did not grow
                std::memcpy(ant.element_pos.colptr(*dest), ant.element_pos.colptr(i_source), 3 * sizeof(dtype));
        }

    qd_arrayant_mark_valid(ant);
}

template void quadriga_lib::copy_element(arrayant<float> &, arma::uword, const arma::uvec &);
template void quadriga_lib::copy_element(arrayant<double> &, arma::uword, const arma::uvec &);

template <typename dtype>
void quadriga_lib::copy_element(std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
                                               arma::uword i_source,
                                               const arma::uvec &i_target)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("Input 'arrayant_vec' cannot be empty.");

    if (i_target.is_empty())
        throw std::invalid_argument("Input 'i_target' cannot be empty.");

    for (size_t i = 0; i < arrayant_vec.size(); ++i)
    {
        try
        {
            quadriga_lib::copy_element(arrayant_vec[i], i_source, i_target);
        }
        catch (const std::invalid_argument &e)
        {
            throw std::invalid_argument("Entry " + std::to_string(i) + ": " + e.what());
        }
    }
}

template void quadriga_lib::copy_element(std::vector<quadriga_lib::arrayant<float>> &, arma::uword, const arma::uvec &);
template void quadriga_lib::copy_element(std::vector<quadriga_lib::arrayant<double>> &, arma::uword, const arma::uvec &);

/*!MD
# set_element_pos
Set element positions for all entries in a multi-frequency arrayant vector

- Writes the same positions to every entry in place, since [[is_valid]] requires equal positions in all entries

## Declaration:
```
void quadriga_lib::set_element_pos(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        const arma::uvec &i_element = arma::uvec());
```

## Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects with equal element counts; updated in place; an empty
  `element_pos` of an entry is filled with zeros first
- **`element_pos`** — New (x, y, z) positions in m; `[3, n_update]` with `n_update` the length of `i_element`, or
  `n_elements` if `i_element` is empty
- **`i_element`** *(optional)* — Indices of the elements to update, 0-based; empty: all elements

## See also:
- [[copy_element]] (adds elements before their positions are set)
- [[join_arrays]] (joins arrays, e.g. the drivers of a loudspeaker)
- [[generate_speaker]] (typical source of multi-frequency arrayant vectors)
MD!*/

template <typename dtype>
void quadriga_lib::set_element_pos(std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
                                                  const arma::Mat<dtype> &element_pos,
                                                  const arma::uvec &i_element)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("Input 'arrayant_vec' cannot be empty.");

    if (element_pos.n_rows != 3)
        throw std::invalid_argument("Input 'element_pos' must have 3 rows (x, y, z).");

    for (const auto &ant : arrayant_vec)
        qd_check_alias(&ant.element_pos, "arrayant_vec", &element_pos);

    arma::uword n_elements_ref = arrayant_vec[0].n_elements();

    // Determine which elements to update
    bool update_all = i_element.is_empty();
    if (update_all)
    {
        // Update all elements: element_pos must have n_elements columns
        if (element_pos.n_cols != n_elements_ref)
            throw std::invalid_argument("Input 'element_pos' has " +
                                        std::to_string(element_pos.n_cols) + " columns but arrayant has " +
                                        std::to_string(n_elements_ref) + " elements.");
    }
    else
    {
        // Update selected elements: element_pos columns must match index count
        if (element_pos.n_cols != i_element.n_elem)
            throw std::invalid_argument("Input 'element_pos' has " +
                                        std::to_string(element_pos.n_cols) + " columns but " +
                                        std::to_string(i_element.n_elem) + " element indices were provided.");

        // Validate indices
        for (arma::uword k = 0; k < i_element.n_elem; ++k)
            if (i_element[k] >= n_elements_ref)
                throw std::invalid_argument("Input 'i_element' contains an index that is out of range.");
    }

    // Ensure the element counts match the reference before any entry is changed
    for (size_t i = 0; i < arrayant_vec.size(); ++i)
        if (arrayant_vec[i].n_elements() != n_elements_ref)
            throw std::invalid_argument("Entry " + std::to_string(i) + ": Number of elements (" + std::to_string(arrayant_vec[i].n_elements()) +
                                        ") does not match the first entry (" + std::to_string(n_elements_ref) + ").");

    // Apply to all entries
    for (size_t i = 0; i < arrayant_vec.size(); ++i)
    {
        quadriga_lib::arrayant<dtype> &ant = arrayant_vec[i];

        // Initialize element_pos if empty
        if (ant.element_pos.is_empty())
            ant.element_pos.zeros(3, n_elements_ref);

        if (update_all)
        {
            ant.element_pos = element_pos;
        }
        else
        {
            for (arma::uword k = 0; k < i_element.n_elem; ++k)
                ant.element_pos.col(i_element[k]) = element_pos.col(k);
        }
    }
}

template void quadriga_lib::set_element_pos(std::vector<quadriga_lib::arrayant<float>> &, const arma::Mat<float> &, const arma::uvec &);
template void quadriga_lib::set_element_pos(std::vector<quadriga_lib::arrayant<double>> &, const arma::Mat<double> &, const arma::uvec &);
