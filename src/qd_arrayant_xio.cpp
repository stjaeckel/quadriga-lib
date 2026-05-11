// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "qd_arrayant_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"
#include "slerp.h"

#include <stdexcept>
#include <filesystem>

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# qdant_write_multi
Write a vector of arrayant objects to a single QDANT file

- Writes each entry in `arrayant_vec` to a QDANT file with sequential 1-based IDs using .[[qdant_write]].
- Auto-generates a `[n_entries, 1]` layout matrix with entries `1, 2, ..., n_entries`.
- Deletes any existing file before writing; all entries are validated first.
- Primary use case: frequency-dependent models where each arrayant holds a pattern at one frequency via `center_frequency`.

## Declaration:
```
void quadriga_lib::qdant_write_multi(
        const std::string &fn,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

## Inputs:
- **`fn`** — Path of the QDANT file to write; must not be empty
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects to store

## See also:
- .[[qdant_write]] (per-object write used internally)
- [[qdant_read]] (read back individual entries by ID)
- [[generate_speaker]] (typical source of frequency-dependent arrayant vectors)
MD!*/

template <typename dtype>
void quadriga_lib::qdant_write_multi(const std::string &fn,
                                     const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("qdant_write_multi: Input vector is empty.");

    if (fn.empty())
        throw std::invalid_argument("qdant_write_multi: Filename cannot be empty.");

    // Remove existing file to start fresh
    if (std::filesystem::exists(fn))
        std::filesystem::remove(fn);

    arma::uword n_entries = (arma::uword)arrayant_vec.size();

    // Validate all entries before writing anything
    for (arma::uword i = 0; i < n_entries; ++i)
    {
        std::string err = arrayant_vec[i].is_valid(false);
        if (!err.empty())
            throw std::invalid_argument("qdant_write_multi: Entry " + std::to_string(i) + " is invalid: " + err);
    }

    // Write all entries without layout (layout validation requires all IDs to exist)
    for (arma::uword i = 0; i < n_entries; ++i)
        arrayant_vec[i].qdant_write(fn, (unsigned)(i + 1));

    // Now all IDs 1..n_entries exist in the file; re-write entry 1 with the full layout
    arma::u32_mat layout(n_entries, 1);
    for (arma::uword i = 0; i < n_entries; ++i)
        layout(i, 0) = (unsigned)(i + 1);

    arrayant_vec[0].qdant_write(fn, 1, layout);
}

template void quadriga_lib::qdant_write_multi(const std::string &, const std::vector<quadriga_lib::arrayant<float>> &);
template void quadriga_lib::qdant_write_multi(const std::string &, const std::vector<quadriga_lib::arrayant<double>> &);

/*!MD
# qdant_read
Read an arrayant object from a QDANT file

- Parses a QuaDRiGa Array Antenna Exchange Format (QDANT) XML file and returns the arrayant for the given ID.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::qdant_read(
        std::string fn,
        unsigned id = 1,
        arma::u32_mat *layout = nullptr);
```

## Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`id`** *(optional)* — 1-based ID of the antenna entry to read
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix of element IDs

## Returns:
- `quadriga_lib::arrayant<dtype>` constructed from the specified entry in the file

## See also:
- .[[qdant_write]] (write a single arrayant)
- [[qdant_write_multi]] (write multiple arrayants with sequential IDs)
MD!*/

// Read array antenna object and layout from QDANT file
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::u32_mat *layout)
{
    quadriga_lib::arrayant<dtype> ant;
    std::string error_message;

    if (layout == nullptr)
    {
        arma::Mat<unsigned> tmp_layout;
        error_message = qd_arrayant_qdant_read(fn, id, &ant.name,
                                               &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                               &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                               &ant.coupling_re, &ant.coupling_im, &ant.center_frequency,
                                               &tmp_layout);
        tmp_layout.reset();
    }
    else
        error_message = qd_arrayant_qdant_read(fn, id, &ant.name,
                                               &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                               &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                               &ant.coupling_re, &ant.coupling_im, &ant.center_frequency,
                                               layout);

    // Throw parsing errors
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Throw validation errors
    error_message = ant.validate();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::Mat<unsigned> *layout);
template quadriga_lib::arrayant<double> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::Mat<unsigned> *layout);

/*!MD
# qdant_read_multi
Read all arrayant objects from a QDANT file into a vector

- Reads all entries from a QDANT file by probing ID 1 to obtain the layout, then reading each unique non-zero ID in order of first appearance (column-major scan).
- Each unique ID is read exactly once regardless of how many times it appears in the layout.
- Counterpart to [[qdant_write_multi]]; primary mechanism for loading frequency-dependent models where `center_frequency` on each entry identifies the corresponding frequency.

## Declaration:
```
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read_multi(
        const std::string &fn,
        arma::u32_mat *layout = nullptr);
```

## Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix; non-zero values are entry IDs

## Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One validated arrayant per unique ID, ordered by first appearance in the layout

## See also:
- [[qdant_read]] (read a single entry by ID)
- [[qdant_write_multi]] (write a vector of arrayants)
- [[generate_speaker]] (typical source of frequency-dependent arrayant vectors)
MD!*/

template <typename dtype>
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read_multi(const std::string &fn, arma::u32_mat *layout)
{
    if (fn.empty())
        throw std::invalid_argument("qdant_read_multi: Filename cannot be empty.");

    // Step 1: Probe the file with id=1 to obtain the layout
    // Even if id=1 does not exist, qd_arrayant_qdant_read still returns the layout
    arma::u32_mat file_layout;
    {
        std::string probe_name;
        arma::Cube<dtype> t_re, t_im, p_re, p_im;
        arma::Col<dtype> az_grid, el_grid;
        arma::Mat<dtype> elem_pos, cpl_re, cpl_im;
        dtype center_freq = dtype(0.0);

        qd_arrayant_qdant_read(fn, 1, &probe_name,
                               &t_re, &t_im, &p_re, &p_im,
                               &az_grid, &el_grid, &elem_pos,
                               &cpl_re, &cpl_im, &center_freq,
                               &file_layout);
        // Error from the probe read is intentionally ignored; we only need the layout.
        // Temporary arrayant data goes out of scope here.
    }

    if (file_layout.is_empty())
        throw std::invalid_argument("qdant_read_multi: Could not read layout from file '" + fn + "'.");

    // Step 2: Extract unique IDs from the layout in order of first appearance
    std::vector<unsigned> unique_ids;
    unique_ids.reserve(file_layout.n_elem);

    for (arma::uword ic = 0; ic < file_layout.n_cols; ++ic)
        for (arma::uword ir = 0; ir < file_layout.n_rows; ++ir)
        {
            unsigned id = file_layout(ir, ic);
            if (id == 0) // Skip empty slots
                continue;

            bool already_seen = false;
            for (size_t k = 0; k < unique_ids.size(); ++k)
                if (unique_ids[k] == id)
                {
                    already_seen = true;
                    break;
                }

            if (!already_seen)
                unique_ids.push_back(id);
        }

    if (unique_ids.empty())
        throw std::invalid_argument("qdant_read_multi: Layout contains no valid (non-zero) IDs in file '" + fn + "'.");

    // Step 3: Read each unique ID
    std::vector<quadriga_lib::arrayant<dtype>> result;
    result.reserve(unique_ids.size());

    for (size_t i = 0; i < unique_ids.size(); ++i)
    {
        int id = (int)unique_ids[i];
        quadriga_lib::arrayant<dtype> ant;
        arma::u32_mat tmp_layout; // Discard layout on subsequent reads

        std::string error_message = qd_arrayant_qdant_read(fn, id, &ant.name,
                                                           &ant.e_theta_re, &ant.e_theta_im,
                                                           &ant.e_phi_re, &ant.e_phi_im,
                                                           &ant.azimuth_grid, &ant.elevation_grid,
                                                           &ant.element_pos,
                                                           &ant.coupling_re, &ant.coupling_im,
                                                           &ant.center_frequency,
                                                           &tmp_layout);

        if (!error_message.empty())
            throw std::invalid_argument("qdant_read_multi: Error reading ID " + std::to_string(id) +
                                        " from '" + fn + "': " + error_message);

        error_message = ant.validate();
        if (!error_message.empty())
            throw std::invalid_argument("qdant_read_multi: Validation failed for ID " + std::to_string(id) +
                                        " from '" + fn + "': " + error_message);

        result.push_back(std::move(ant));
    }

    // Step 4: Optionally return the layout
    if (layout != nullptr)
        *layout = file_layout;

    return result;
}

template std::vector<quadriga_lib::arrayant<float>> quadriga_lib::qdant_read_multi(const std::string &, arma::u32_mat *);
template std::vector<quadriga_lib::arrayant<double>> quadriga_lib::qdant_read_multi(const std::string &, arma::u32_mat *);

/*!MD
# arrayant_is_valid_multi
Validate a vector of arrayant objects for multi-frequency consistency

- Each entry is validated individually via its `is_valid` member; `quick_check` is forwarded to that call.
- Cross-entry checks (all vs. entry 0): azimuth/elevation grid sizes and values, number of elements, element positions, coupling_re shape, and coupling_im presence and size.
- Pattern data, `center_frequency`, and coupling matrix values are not compared (expected to vary).
- Stops at first error and returns a message identifying the failing entry and property.

## Declaration:
```
std::string quadriga_lib::arrayant_is_valid_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        bool quick_check = true);
```

## Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects to validate
- **`quick_check`** *(optional)* — If `true`, uses fast pointer-based per-entry validation; if `false`, performs full deep validation

## Returns:
- Empty string if valid; otherwise a message such as `"Entry 3: Azimuth grid values do not match entry 0."`

## See also:
- .[[is_valid]] (per-entry validation called internally)
- [[generate_speaker]] (typical source of multi-frequency arrayant vectors)
MD!*/

template <typename dtype>
std::string quadriga_lib::arrayant_is_valid_multi(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec, bool quick_check)
{
    if (arrayant_vec.empty())
        return std::string("Input vector is empty.");

    // Validate first entry (always deep-check the reference entry)
    std::string err = arrayant_vec[0].is_valid(quick_check);
    if (!err.empty())
        return std::string("Entry 0: ") + err;

    // Reference dimensions and pointers from the first entry
    arma::uword n_el_ref = arrayant_vec[0].n_elevation();
    arma::uword n_az_ref = arrayant_vec[0].n_azimuth();
    arma::uword n_elements_ref = arrayant_vec[0].n_elements();
    arma::uword n_cpl_rows_ref = arrayant_vec[0].coupling_re.n_rows;
    arma::uword n_cpl_cols_ref = arrayant_vec[0].coupling_re.n_cols;
    bool has_element_pos_ref = !arrayant_vec[0].element_pos.empty();
    bool has_coupling_im_ref = !arrayant_vec[0].coupling_im.empty();

    for (size_t i = 1; i < arrayant_vec.size(); ++i)
    {
        std::string idx = std::to_string(i);

        // Validate individual entry
        err = arrayant_vec[i].is_valid(quick_check);
        if (!err.empty())
            return std::string("Entry ") + idx + ": " + err;

        // Check angular grid sizes
        if (arrayant_vec[i].n_elevation() != n_el_ref)
            return std::string("Entry ") + idx + ": Number of elevation angles (" +
                   std::to_string(arrayant_vec[i].n_elevation()) + ") does not match entry 0 (" +
                   std::to_string(n_el_ref) + ").";

        if (arrayant_vec[i].n_azimuth() != n_az_ref)
            return std::string("Entry ") + idx + ": Number of azimuth angles (" +
                   std::to_string(arrayant_vec[i].n_azimuth()) + ") does not match entry 0 (" +
                   std::to_string(n_az_ref) + ").";

        // Check angular grid values
        const dtype *az_ref = arrayant_vec[0].azimuth_grid.memptr();
        const dtype *az_cur = arrayant_vec[i].azimuth_grid.memptr();
        for (arma::uword k = 0; k < n_az_ref; ++k)
            if (az_ref[k] != az_cur[k])
                return std::string("Entry ") + idx + ": Azimuth grid values do not match entry 0.";

        const dtype *el_ref = arrayant_vec[0].elevation_grid.memptr();
        const dtype *el_cur = arrayant_vec[i].elevation_grid.memptr();
        for (arma::uword k = 0; k < n_el_ref; ++k)
            if (el_ref[k] != el_cur[k])
                return std::string("Entry ") + idx + ": Elevation grid values do not match entry 0.";

        // Check number of elements
        if (arrayant_vec[i].n_elements() != n_elements_ref)
            return std::string("Entry ") + idx + ": Number of elements (" +
                   std::to_string(arrayant_vec[i].n_elements()) + ") does not match entry 0 (" +
                   std::to_string(n_elements_ref) + ").";

        // Check element positions
        bool has_element_pos_cur = !arrayant_vec[i].element_pos.empty();
        if (has_element_pos_ref != has_element_pos_cur)
            return std::string("Entry ") + idx + ": Presence of element_pos does not match entry 0.";

        if (has_element_pos_ref && has_element_pos_cur)
        {
            if (arrayant_vec[i].element_pos.n_rows != arrayant_vec[0].element_pos.n_rows ||
                arrayant_vec[i].element_pos.n_cols != arrayant_vec[0].element_pos.n_cols)
                return std::string("Entry ") + idx + ": Size of element_pos does not match entry 0.";

            const dtype *pos_ref = arrayant_vec[0].element_pos.memptr();
            const dtype *pos_cur = arrayant_vec[i].element_pos.memptr();
            arma::uword n_pos = arrayant_vec[0].element_pos.n_elem;
            for (arma::uword k = 0; k < n_pos; ++k)
                if (pos_ref[k] != pos_cur[k])
                    return std::string("Entry ") + idx + ": Element positions do not match entry 0.";
        }

        // Check coupling matrix shape
        if (arrayant_vec[i].coupling_re.n_rows != n_cpl_rows_ref ||
            arrayant_vec[i].coupling_re.n_cols != n_cpl_cols_ref)
            return std::string("Entry ") + idx + ": Coupling matrix size (" +
                   std::to_string(arrayant_vec[i].coupling_re.n_rows) + "x" +
                   std::to_string(arrayant_vec[i].coupling_re.n_cols) + ") does not match entry 0 (" +
                   std::to_string(n_cpl_rows_ref) + "x" + std::to_string(n_cpl_cols_ref) + ").";

        bool has_coupling_im_cur = !arrayant_vec[i].coupling_im.empty();
        if (has_coupling_im_ref != has_coupling_im_cur)
            return std::string("Entry ") + idx + ": Presence of coupling_im does not match entry 0.";

        if (has_coupling_im_ref && has_coupling_im_cur)
            if (arrayant_vec[i].coupling_im.n_rows != arrayant_vec[0].coupling_im.n_rows ||
                arrayant_vec[i].coupling_im.n_cols != arrayant_vec[0].coupling_im.n_cols)
                return std::string("Entry ") + idx + ": Size of coupling_im does not match entry 0.";
    }

    return std::string("");
}

template std::string quadriga_lib::arrayant_is_valid_multi(const std::vector<quadriga_lib::arrayant<float>> &, bool);
template std::string quadriga_lib::arrayant_is_valid_multi(const std::vector<quadriga_lib::arrayant<double>> &, bool);

/*!MD
# arrayant_copy_element_multi
Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector

- Calls .[[copy_element]] on every entry in the vector with the same source and destination indices.
- If any destination index exceeds the current element count, all entries are enlarged; new elements receive an identity coupling entry.
- Source and destination indices are 0-based.

## Declaration:
```
void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uvec destination);

void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uword destination);
```

## Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects; modified in-place
- **`source`** — Index of the element to copy from; must be within current element count
- **`destination`** — Index or indices of target elements; enlarges all entries if any index exceeds current count

## Example:
```
arma::vec freqs = {500.0, 1000.0, 2000.0, 5000.0};
auto driver = quadriga_lib::generate_speaker<double>(
    "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
    0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
quadriga_lib::arrayant_copy_element_multi(driver, 0, arma::uvec{1, 2, 3});
```

## See also:
- .[[copy_element]] (per-entry operation called internally)
- [[arrayant_set_element_pos_multi]] (set element positions after copying)
- [[arrayant_concat_multi]] (combine multiple arrayant vectors)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_copy_element_multi(std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
                                               arma::uword source,
                                               arma::uvec destination)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("arrayant_copy_element_multi: Input vector is empty.");

    if (destination.is_empty())
        throw std::invalid_argument("arrayant_copy_element_multi: Destination index vector is empty.");

    for (size_t i = 0; i < arrayant_vec.size(); ++i)
    {
        try
        {
            arrayant_vec[i].copy_element(source, destination);
        }
        catch (const std::invalid_argument &e)
        {
            throw std::invalid_argument("arrayant_copy_element_multi: Error at entry " +
                                        std::to_string(i) + ": " + e.what());
        }
    }
}

template <typename dtype>
void quadriga_lib::arrayant_copy_element_multi(std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
                                               arma::uword source,
                                               arma::uword destination)
{
    arma::uvec dest(1);
    dest.at(0) = destination;
    quadriga_lib::arrayant_copy_element_multi(arrayant_vec, source, dest);
}

template void quadriga_lib::arrayant_copy_element_multi(std::vector<quadriga_lib::arrayant<float>> &, arma::uword, arma::uvec);
template void quadriga_lib::arrayant_copy_element_multi(std::vector<quadriga_lib::arrayant<float>> &, arma::uword, arma::uword);
template void quadriga_lib::arrayant_copy_element_multi(std::vector<quadriga_lib::arrayant<double>> &, arma::uword, arma::uvec);
template void quadriga_lib::arrayant_copy_element_multi(std::vector<quadriga_lib::arrayant<double>> &, arma::uword, arma::uword);

/*!MD
# arrayant_set_element_pos_multi
Set element positions for all entries in a multi-frequency arrayant vector

- Updates `element_pos` in-place on every entry in the vector identically.
- If `i_element` is empty, all positions are replaced and `element_pos` must have `n_elements` columns.
- If `i_element` is provided, only those indexed columns are updated; `element_pos` column count must match `i_element` length.
- All entries must have the same element count; uninitialized `element_pos` fields are zero-initialized before update.

## Declaration:
```
void quadriga_lib::arrayant_set_element_pos_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        arma::uvec i_element = arma::uvec());
```

## Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`element_pos`** — New (x, y, z) positions; `[3, n_update]`
- **`i_element`** *(optional)* — Indices of elements to update; if empty, all elements are replaced

## See also:
- [[arrayant_copy_element_multi]] (replicate elements before setting positions)
- [[generate_speaker]] (typical source of multi-frequency arrayant vectors)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_set_element_pos_multi(std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
                                                  const arma::Mat<dtype> &element_pos,
                                                  arma::uvec i_element)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("arrayant_set_element_pos_multi: Input vector is empty.");

    if (element_pos.n_rows != 3)
        throw std::invalid_argument("arrayant_set_element_pos_multi: element_pos must have 3 rows (x, y, z).");

    arma::uword n_elements_ref = arrayant_vec[0].n_elements();

    // Determine which elements to update
    bool update_all = i_element.is_empty();
    if (update_all)
    {
        // Update all elements: element_pos must have n_elements columns
        if (element_pos.n_cols != n_elements_ref)
            throw std::invalid_argument("arrayant_set_element_pos_multi: element_pos has " +
                                        std::to_string(element_pos.n_cols) + " columns but arrayant has " +
                                        std::to_string(n_elements_ref) + " elements.");
    }
    else
    {
        // Update selected elements: element_pos columns must match index count
        if (element_pos.n_cols != i_element.n_elem)
            throw std::invalid_argument("arrayant_set_element_pos_multi: element_pos has " +
                                        std::to_string(element_pos.n_cols) + " columns but " +
                                        std::to_string(i_element.n_elem) + " element indices were provided.");

        // Validate indices
        for (arma::uword k = 0; k < i_element.n_elem; ++k)
            if (i_element[k] >= n_elements_ref)
                throw std::invalid_argument("arrayant_set_element_pos_multi: Element index " +
                                            std::to_string(i_element[k]) + " is out of range (n_elements = " +
                                            std::to_string(n_elements_ref) + ").");
    }

    // Apply to all entries
    for (size_t i = 0; i < arrayant_vec.size(); ++i)
    {
        quadriga_lib::arrayant<dtype> &ant = arrayant_vec[i];

        // Ensure element count matches the reference
        if (ant.n_elements() != n_elements_ref)
            throw std::invalid_argument("arrayant_set_element_pos_multi: Entry " + std::to_string(i) +
                                        " has " + std::to_string(ant.n_elements()) +
                                        " elements, expected " + std::to_string(n_elements_ref) + ".");

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

template void quadriga_lib::arrayant_set_element_pos_multi(std::vector<quadriga_lib::arrayant<float>> &, const arma::Mat<float> &, arma::uvec);
template void quadriga_lib::arrayant_set_element_pos_multi(std::vector<quadriga_lib::arrayant<double>> &, const arma::Mat<double> &, arma::uvec);

/*!MD
# arrayant_rotate_pattern_multi
Apply Euler rotations to all entries in a multi-frequency arrayant vector

- Calls .[[rotate_pattern]] on every entry with grid adjustment always disabled (required for uniform-grid consistency across frequencies).
- If `i_element` is empty, all elements are rotated; otherwise only the specified indices are affected.
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid spurious polarization effects.

## Declaration:
```
void quadriga_lib::arrayant_rotate_pattern_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        dtype x_deg = 0.0,
        dtype y_deg = 0.0,
        dtype z_deg = 0.0,
        unsigned usage = 0,
        arma::uvec i_element = arma::uvec())
```

## Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`x_deg`** *(optional)* — Bank angle in degrees
- **`y_deg`** *(optional)* — Tilt angle in degrees
- **`z_deg`** *(optional)* — Heading angle in degrees
- **`usage`** *(optional)* — Rotation mode: `0` = pattern + polarization, `1` = pattern only, `2` = polarization only
- **`i_element`** *(optional)* — Indices of elements to rotate; if empty, all elements are rotated

## See also:
- .[[rotate_pattern]] (per-entry operation called internally)
- [[arrayant_concat_multi]] (combine multi-frequency vectors before rotating)
- [[arrayant_set_element_pos_multi]] (set element positions in multi-frequency vectors)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_rotate_pattern_multi(std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
                                                 dtype x_deg, dtype y_deg, dtype z_deg,
                                                 unsigned usage, arma::uvec i_element)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("arrayant_rotate_pattern_multi: Input vector is empty.");

    if (usage > 2)
        throw std::invalid_argument("arrayant_rotate_pattern_multi: Usage must be 0, 1, or 2.");

    // Validate input vector before modifying any entries
    std::string err = quadriga_lib::arrayant_is_valid_multi(arrayant_vec, true);
    if (!err.empty())
        throw std::invalid_argument("arrayant_rotate_pattern_multi: Input validation failed: " + err);

    // Map usage to no-grid-adjustment modes: 0→3, 1→4, 2→2
    unsigned internal_usage = usage;
    if (usage == 0)
        internal_usage = 3;
    else if (usage == 1)
        internal_usage = 4;

    // Determine element list
    bool rotate_all = i_element.is_empty();

    // Validate element indices against the first entry
    if (!rotate_all)
    {
        arma::uword n_elements_ref = arrayant_vec[0].n_elements();
        for (arma::uword k = 0; k < i_element.n_elem; ++k)
            if (i_element[k] >= n_elements_ref)
                throw std::invalid_argument("arrayant_rotate_pattern_multi: Element index " +
                                            std::to_string(i_element[k]) + " is out of range (n_elements = " +
                                            std::to_string(n_elements_ref) + ").");
    }

    for (size_t i = 0; i < arrayant_vec.size(); ++i)
    {
        try
        {
            if (rotate_all)
            {
                // rotate_pattern with element = -1 rotates all elements
                arrayant_vec[i].rotate_pattern(x_deg, y_deg, z_deg, internal_usage);
            }
            else
            {
                // Rotate each specified element individually
                for (arma::uword k = 0; k < i_element.n_elem; ++k)
                    arrayant_vec[i].rotate_pattern(x_deg, y_deg, z_deg, internal_usage, (unsigned)i_element[k]);
            }
        }
        catch (const std::invalid_argument &e)
        {
            throw std::invalid_argument("arrayant_rotate_pattern_multi: Error at entry " +
                                        std::to_string(i) + ": " + e.what());
        }
    }
}

template void quadriga_lib::arrayant_rotate_pattern_multi(std::vector<quadriga_lib::arrayant<float>> &, float, float, float, unsigned, arma::uvec);
template void quadriga_lib::arrayant_rotate_pattern_multi(std::vector<quadriga_lib::arrayant<double>> &, double, double, double, unsigned, arma::uvec);

/*!MD
# arrayant_concat_multi
Concatenate two multi-frequency arrayant vectors into a single multi-element model

- Both inputs must have equal entry counts, identical angular grids, and matching `center_frequency` values at each index.
- Per frequency entry: pattern cubes are joined along the element (slice) dimension; `element_pos` matrices are horizontally concatenated (empty positions treated as zeros).
- Both inputs are validated with [[arrayant_is_valid_multi]] before processing; each output entry is validated before returning.
- Output inherits name, azimuth/elevation grids, and `center_frequency` from `arrayant_vec1`.

## Declaration:
```
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::arrayant_concat_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec1,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec2);
```

## Inputs:
- **`arrayant_vec1`** — First validated, mutually consistent arrayant vector
- **`arrayant_vec2`** — Second arrayant vector; must match entry count, grids, and center frequencies of `arrayant_vec1`

## Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports per entry
- Coupling matrices are assembled block-diagonally — elements from `vec1` connect only to ports from `vec1` and vice versa:<br><br>
   | Element \ Port      | P1…Pp1 (vec1) | Pp1+1…Pp1+p2 (vec2) |
   | ------------------- | :-----------: | :-----------------: |
   | E1…En1 (vec1)       |   C1 block    |          0          |
   | En1+1…En1+n2 (vec2) |       0       |      C2 block       |

## See also:
- [[arrayant_is_valid_multi]] (validation called on both inputs)
- [[arrayant_set_element_pos_multi]] (position drivers before concatenating)
- [[arrayant_rotate_pattern_multi]] (rotate elements after concatenating)
- [[qdant_write_multi]] (persist the combined model)
MD!*/

template <typename dtype>
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::arrayant_concat_multi(
    const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec1,
    const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec2)
{
    // --- Validate individual inputs ---
    std::string err = quadriga_lib::arrayant_is_valid_multi(arrayant_vec1, false);
    if (!err.empty())
        throw std::invalid_argument("arrayant_concat_multi: First input invalid: " + err);

    err = quadriga_lib::arrayant_is_valid_multi(arrayant_vec2, false);
    if (!err.empty())
        throw std::invalid_argument("arrayant_concat_multi: Second input invalid: " + err);

    // --- Check frequency count ---
    arma::uword n_freq = (arma::uword)arrayant_vec1.size();
    if ((arma::uword)arrayant_vec2.size() != n_freq)
        throw std::invalid_argument("arrayant_concat_multi: Inputs have different number of frequency entries (" +
                                    std::to_string(arrayant_vec1.size()) + " vs " +
                                    std::to_string(arrayant_vec2.size()) + ").");

    // --- Check angular grid compatibility (compare against first entry of vec1) ---
    arma::uword n_el = arrayant_vec1[0].n_elevation();
    arma::uword n_az = arrayant_vec1[0].n_azimuth();

    if (arrayant_vec2[0].n_elevation() != n_el)
        throw std::invalid_argument("arrayant_concat_multi: Elevation grid sizes do not match (" +
                                    std::to_string(n_el) + " vs " +
                                    std::to_string(arrayant_vec2[0].n_elevation()) + ").");

    if (arrayant_vec2[0].n_azimuth() != n_az)
        throw std::invalid_argument("arrayant_concat_multi: Azimuth grid sizes do not match (" +
                                    std::to_string(n_az) + " vs " +
                                    std::to_string(arrayant_vec2[0].n_azimuth()) + ").");

    const dtype *az_ref = arrayant_vec1[0].azimuth_grid.memptr();
    const dtype *az_cur = arrayant_vec2[0].azimuth_grid.memptr();
    for (arma::uword k = 0; k < n_az; ++k)
        if (az_ref[k] != az_cur[k])
            throw std::invalid_argument("arrayant_concat_multi: Azimuth grid values do not match.");

    const dtype *el_ref = arrayant_vec1[0].elevation_grid.memptr();
    const dtype *el_cur = arrayant_vec2[0].elevation_grid.memptr();
    for (arma::uword k = 0; k < n_el; ++k)
        if (el_ref[k] != el_cur[k])
            throw std::invalid_argument("arrayant_concat_multi: Elevation grid values do not match.");

    // --- Reference dimensions ---
    arma::uword n_elem1 = arrayant_vec1[0].n_elements();
    arma::uword n_elem2 = arrayant_vec2[0].n_elements();
    arma::uword n_ports1 = arrayant_vec1[0].coupling_re.n_cols;
    arma::uword n_ports2 = arrayant_vec2[0].coupling_re.n_cols;
    bool has_coupling_im1 = !arrayant_vec1[0].coupling_im.empty();
    bool has_coupling_im2 = !arrayant_vec2[0].coupling_im.empty();

    // --- Build output ---
    std::vector<quadriga_lib::arrayant<dtype>> output(n_freq);

    for (arma::uword i = 0; i < n_freq; ++i)
    {
        const quadriga_lib::arrayant<dtype> &a1 = arrayant_vec1[i];
        const quadriga_lib::arrayant<dtype> &a2 = arrayant_vec2[i];

        // Check center frequency match
        if (std::abs((double)a1.center_frequency - (double)a2.center_frequency) > 1.0e-6 * std::abs((double)a1.center_frequency))
            throw std::invalid_argument("arrayant_concat_multi: Center frequency mismatch at index " +
                                        std::to_string(i) + " (" + std::to_string((double)a1.center_frequency) +
                                        " vs " + std::to_string((double)a2.center_frequency) + ").");

        quadriga_lib::arrayant<dtype> &out = output[i];

        // Copy grids and metadata from vec1
        out.name = a1.name;
        out.azimuth_grid = a1.azimuth_grid;
        out.elevation_grid = a1.elevation_grid;
        out.center_frequency = a1.center_frequency;

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
        arma::uword n_elem_out = n_elem1 + n_elem2;
        arma::uword n_ports_out = n_ports1 + n_ports2;

        out.coupling_re.zeros(n_elem_out, n_ports_out);
        out.coupling_re.submat(0, 0, n_elem1 - 1, n_ports1 - 1) = a1.coupling_re;
        out.coupling_re.submat(n_elem1, n_ports1, n_elem_out - 1, n_ports_out - 1) = a2.coupling_re;

        if (has_coupling_im1 || has_coupling_im2)
        {
            out.coupling_im.zeros(n_elem_out, n_ports_out);
            if (has_coupling_im1)
                out.coupling_im.submat(0, 0, n_elem1 - 1, n_ports1 - 1) = a1.coupling_im;
            if (has_coupling_im2)
                out.coupling_im.submat(n_elem1, n_ports1, n_elem_out - 1, n_ports_out - 1) = a2.coupling_im;
        }

        // Validate and initialize check_ptr
        err = out.validate();
        if (!err.empty())
            throw std::invalid_argument("arrayant_concat_multi: Output validation failed at index " +
                                        std::to_string(i) + ": " + err);
    }

    return output;
}

template std::vector<quadriga_lib::arrayant<float>> quadriga_lib::arrayant_concat_multi(
    const std::vector<quadriga_lib::arrayant<float>> &, const std::vector<quadriga_lib::arrayant<float>> &);

template std::vector<quadriga_lib::arrayant<double>> quadriga_lib::arrayant_concat_multi(
    const std::vector<quadriga_lib::arrayant<double>> &, const std::vector<quadriga_lib::arrayant<double>> &);

/*!MD
# arrayant_combine_pattern_multi
Combine element patterns, positions, and coupling weights into effective radiation patterns (multi-frequency)

- Multi-frequency counterpart to .[[combine_pattern]]
- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` across all entries to produce one output element per port (column of the coupling matrix) at each requested output frequency
- Output length = `freq_grid_new->n_elem` if provided; otherwise one entry per input arrayant
- Field interpolation across frequency is delegated to [[arrayant_interpolate_multi]] (SLERP with linear-interpolation fallback)
- Coupling matrices are SLERP-interpolated between bracketing input entries; out-of-range output frequencies are clamped to the nearest input entry
- Each output arrayant has identity coupling and zero element positions (patterns are pre-combined)

## Declaration:
```
std::vector<arrayant<dtype>> arrayant_combine_pattern_multi(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr,
    const arma::Col<dtype> *freq_grid_new = nullptr);
```

## Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects (must pass [[arrayant_is_valid_multi]])
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], sorted; defaults to `arrayant_vec[0].azimuth_grid`
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], sorted; defaults to `arrayant_vec[0].elevation_grid`
- **`freq_grid_new`** *(optional)* — Alternative frequency grid in Hz; defaults to per-entry `center_frequency`

## Returns:
- Vector of arrayant objects (length = `n_freq_out`), each with `n_ports` elements equal to the number of columns in the coupling matrix

## See also:
- .[[combine_pattern]] (single-frequency counterpart)
- [[arrayant_interpolate_multi]] (used internally for spatial+frequency field interpolation)
- [[arrayant_is_valid_multi]] (input validation)
MD!*/

template <typename dtype>
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::arrayant_combine_pattern_multi(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Col<dtype> *azimuth_grid_new,
    const arma::Col<dtype> *elevation_grid_new,
    const arma::Col<dtype> *freq_grid_new)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("arrayant_combine_pattern_multi: Input vector is empty.");

    // Validate input vector (deep check)
    std::string err = quadriga_lib::arrayant_is_valid_multi(arrayant_vec, false);
    if (!err.empty())
        throw std::invalid_argument("arrayant_combine_pattern_multi: Input validation failed: " + err);

    arma::uword n_freq_in = (arma::uword)arrayant_vec.size();

    // Validate optional output grids
    bool new_azgrid = azimuth_grid_new != nullptr && azimuth_grid_new->n_elem != 0;
    bool new_elgrid = elevation_grid_new != nullptr && elevation_grid_new->n_elem != 0;
    bool new_fgrid = freq_grid_new != nullptr && freq_grid_new->n_elem != 0;

    if (new_azgrid && !qd_in_range(azimuth_grid_new->memptr(), azimuth_grid_new->n_elem, dtype(-3.1415930), dtype(3.1415930), true, true))
        throw std::invalid_argument("arrayant_combine_pattern_multi: Values of 'azimuth_grid_new' must be sorted and in [-pi, pi].");

    if (new_elgrid && !qd_in_range(elevation_grid_new->memptr(), elevation_grid_new->n_elem, dtype(-1.5707965), dtype(1.5707965), true, true))
        throw std::invalid_argument("arrayant_combine_pattern_multi: Values of 'elevation_grid_new' must be sorted and in [-pi/2, pi/2].");

    if (new_fgrid)
    {
        const dtype *p_freq = freq_grid_new->memptr();
        for (arma::uword i = 0; i < freq_grid_new->n_elem; ++i)
            if (!(p_freq[i] > dtype(0.0)))
                throw std::invalid_argument("arrayant_combine_pattern_multi: Values of 'freq_grid_new' must be positive.");
    }

    // Output grids
    const dtype *p_azimuth_grid = new_azgrid ? azimuth_grid_new->memptr() : arrayant_vec[0].azimuth_grid.memptr();
    const dtype *p_elevation_grid = new_elgrid ? elevation_grid_new->memptr() : arrayant_vec[0].elevation_grid.memptr();
    arma::uword n_azimuth_out = new_azgrid ? azimuth_grid_new->n_elem : arrayant_vec[0].azimuth_grid.n_elem;
    arma::uword n_elevation_out = new_elgrid ? elevation_grid_new->n_elem : arrayant_vec[0].elevation_grid.n_elem;
    arma::uword n_ang = n_azimuth_out * n_elevation_out;

    // Output frequency grid: default = per-entry center_frequency in input order
    arma::Col<dtype> freq_grid;
    if (new_fgrid)
        freq_grid = *freq_grid_new;
    else
    {
        freq_grid.set_size(n_freq_in);
        for (arma::uword i = 0; i < n_freq_in; ++i)
            freq_grid(i) = arrayant_vec[i].center_frequency;
    }
    arma::uword n_freq_out = freq_grid.n_elem;

    // Build flat (azimuth, elevation) pair list (elevation varies fastest)
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    {
        dtype *p_az = azimuth.memptr(), *p_el = elevation.memptr();
        for (arma::uword ia = 0; ia < n_azimuth_out; ++ia)
            for (arma::uword ie = 0; ie < n_elevation_out; ++ie)
                *p_az++ = p_azimuth_grid[ia], *p_el++ = p_elevation_grid[ie];
    }

    // Element positions (identical across input entries per arrayant_is_valid_multi)
    arma::uword n_elements = arrayant_vec[0].n_elements();
    arma::Mat<dtype> element_pos_used = arrayant_vec[0].element_pos.is_empty()
                                            ? arma::Mat<dtype>(3, n_elements, arma::fill::zeros)
                                            : arrayant_vec[0].element_pos;

    // Interpolate fields across angles and freq; pass zero element_pos so no phase is baked in
    arma::Mat<dtype> zero_pos(3, n_elements, arma::fill::zeros);
    arma::Cube<dtype> V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi<dtype>(arrayant_vec, &azimuth, &elevation, &freq_grid,
                                                    &V_re, &V_im, &H_re, &H_im, arma::uvec{}, nullptr, &zero_pos, false);

    // Geometric dist matrix: dist(e, a) = element_pos[:,e] · wave_direction(az[a], el[a])
    arma::Mat<dtype> dist(n_elements, n_ang, arma::fill::none);
    {
        const dtype *p_pos = element_pos_used.memptr();
        const dtype *p_az = azimuth.memptr();
        const dtype *p_el = elevation.memptr();
        dtype *p_dist = dist.memptr(); // column-major: dist(e, a)
        for (arma::uword a = 0; a < n_ang; ++a)
        {
            dtype caz = std::cos(p_az[a]), saz = std::sin(p_az[a]);
            dtype cel = std::cos(p_el[a]), sel = std::sin(p_el[a]);
            dtype dx = cel * caz, dy = cel * saz, dz = sel;
            for (arma::uword e = 0; e < n_elements; ++e)
            {
                dtype x = p_pos[3 * e + 0], y = p_pos[3 * e + 1], z = p_pos[3 * e + 2];
                *p_dist++ = -(x * dx + y * dy + z * dz);
            }
        }
    }

    // Coupling setup
    bool no_coupling = arrayant_vec[0].coupling_re.is_empty();
    arma::uword n_ports_out = no_coupling ? n_elements : arrayant_vec[0].coupling_re.n_cols;

    // Input center frequencies and sort order (for coupling interpolation)
    arma::Col<double> f_in_d(n_freq_in);
    for (arma::uword i = 0; i < n_freq_in; ++i)
        f_in_d(i) = (double)arrayant_vec[i].center_frequency;
    arma::uvec sort_idx = arma::sort_index(f_in_d);
    arma::Col<double> f_in_sorted(n_freq_in);
    for (arma::uword i = 0; i < n_freq_in; ++i)
        f_in_sorted(i) = f_in_d(sort_idx(i));

    // Build output entries
    std::vector<quadriga_lib::arrayant<dtype>> output(n_freq_out);
    constexpr double C0 = 299792458.0;

    for (arma::uword f = 0; f < n_freq_out; ++f)
    {
        double freq_out = (double)freq_grid(f);
        dtype wave_no = dtype(2.0 * arma::datum::pi * freq_out / C0);

        // Coupling for this output frequency (linear interp of complex coupling, clamped)
        arma::Mat<std::complex<dtype>> coupling;
        if (no_coupling)
        {
            coupling = arma::Mat<std::complex<dtype>>(n_elements, n_ports_out, arma::fill::zeros);
            for (arma::uword k = 0; k < n_elements; ++k)
                coupling(k, k) = std::complex<dtype>(dtype(1.0), dtype(0.0));
        }
        else
        {
            arma::uword i_lo = sort_idx(0), i_hi = sort_idx(0);
            double w = 0.0;
            if (freq_out <= f_in_sorted(0))
                i_lo = i_hi = sort_idx(0), w = 0.0;
            else if (freq_out >= f_in_sorted(n_freq_in - 1))
                i_lo = i_hi = sort_idx(n_freq_in - 1), w = 0.0;
            else
                for (arma::uword i = 1; i < n_freq_in; ++i)
                    if (f_in_sorted(i) >= freq_out)
                    {
                        i_lo = sort_idx(i - 1);
                        i_hi = sort_idx(i);
                        double f_lo = f_in_sorted(i - 1), f_hi = f_in_sorted(i);
                        w = (freq_out - f_lo) / (f_hi - f_lo);
                        break;
                    }

            const arma::Mat<dtype> &cre_lo = arrayant_vec[i_lo].coupling_re;
            const arma::Mat<dtype> &cre_hi = arrayant_vec[i_hi].coupling_re;
            arma::Mat<dtype> cim_lo = arrayant_vec[i_lo].coupling_im.is_empty()
                                          ? arma::Mat<dtype>(n_elements, n_ports_out, arma::fill::zeros)
                                          : arrayant_vec[i_lo].coupling_im;
            arma::Mat<dtype> cim_hi = arrayant_vec[i_hi].coupling_im.is_empty()
                                          ? arma::Mat<dtype>(n_elements, n_ports_out, arma::fill::zeros)
                                          : arrayant_vec[i_hi].coupling_im;

            arma::Mat<dtype> cre(n_elements, n_ports_out);
            arma::Mat<dtype> cim(n_elements, n_ports_out);
            const dtype *p_rlo = cre_lo.memptr(), *p_rhi = cre_hi.memptr();
            const dtype *p_ilo = cim_lo.memptr(), *p_ihi = cim_hi.memptr();
            dtype *p_ro = cre.memptr(), *p_io = cim.memptr();
            const arma::uword nE = n_elements * n_ports_out;
            for (arma::uword k = 0; k < nE; ++k)
                slerp_complex_mf(p_rlo[k], p_ilo[k], p_rhi[k], p_ihi[k], dtype(w), p_ro[k], p_io[k]);
            coupling = arma::Mat<std::complex<dtype>>(cre, cim);
        }

        // Apply element-position phase shift: exp(-j * k * dist)
        arma::Mat<dtype> wd = wave_no * dist; // [n_elements, n_ang]
        arma::Mat<std::complex<dtype>> phase(arma::cos(wd), arma::sin(-wd));

        arma::Mat<dtype> V_re_s = V_re.slice(f); // [n_elements, n_ang]
        arma::Mat<dtype> V_im_s = V_im.slice(f);
        arma::Mat<dtype> H_re_s = H_re.slice(f);
        arma::Mat<dtype> H_im_s = H_im.slice(f);
        arma::Mat<std::complex<dtype>> Vi(V_re_s, V_im_s);
        arma::Mat<std::complex<dtype>> Hi(H_re_s, H_im_s);
        Vi = Vi % phase;
        Hi = Hi % phase;

        // Apply coupling: Vo(ang, port) = sum_e coupling(e, port) * Vi(e, ang)
        arma::Mat<std::complex<dtype>> Vo(n_ang, n_ports_out, arma::fill::zeros);
        arma::Mat<std::complex<dtype>> Ho(n_ang, n_ports_out, arma::fill::zeros);
        for (arma::uword i = 0; i < n_elements; ++i)
        {
            arma::Col<std::complex<dtype>> vi = Vi.row(i).as_col();
            arma::Col<std::complex<dtype>> hi = Hi.row(i).as_col();
            for (arma::uword o = 0; o < n_ports_out; ++o)
            {
                std::complex<dtype> cpl = coupling.at(i, o);
                Vo.col(o) += vi * cpl;
                Ho.col(o) += hi * cpl;
            }
        }

        // Write output entry
        quadriga_lib::arrayant<dtype> &out = output[f];
        out.set_size(n_elevation_out, n_azimuth_out, n_ports_out, n_ports_out);

        std::memcpy(out.azimuth_grid.memptr(), p_azimuth_grid, n_azimuth_out * sizeof(dtype));
        std::memcpy(out.elevation_grid.memptr(), p_elevation_grid, n_elevation_out * sizeof(dtype));

        arma::uword n_bytes = n_elevation_out * n_azimuth_out * n_ports_out * sizeof(dtype);
        arma::Mat<dtype> cpy = arma::real(Vo);
        std::memcpy(out.e_theta_re.memptr(), cpy.memptr(), n_bytes);
        cpy = arma::imag(Vo);
        std::memcpy(out.e_theta_im.memptr(), cpy.memptr(), n_bytes);
        cpy = arma::real(Ho);
        std::memcpy(out.e_phi_re.memptr(), cpy.memptr(), n_bytes);
        cpy = arma::imag(Ho);
        std::memcpy(out.e_phi_im.memptr(), cpy.memptr(), n_bytes);

        out.element_pos.zeros();
        out.coupling_re.eye();
        out.coupling_im.zeros();
        out.center_frequency = dtype(freq_out);
        out.name = arrayant_vec[0].name;

        // Quick-check pointers
        out.check_ptr[0] = out.e_theta_re.memptr();
        out.check_ptr[1] = out.e_theta_im.memptr();
        out.check_ptr[2] = out.e_phi_re.memptr();
        out.check_ptr[3] = out.e_phi_im.memptr();
        out.check_ptr[4] = out.azimuth_grid.memptr();
        out.check_ptr[5] = out.elevation_grid.memptr();
        out.check_ptr[6] = out.element_pos.memptr();
        out.check_ptr[7] = out.coupling_re.memptr();
        out.check_ptr[8] = out.coupling_im.memptr();
    }

    return output;
}

template std::vector<quadriga_lib::arrayant<float>> quadriga_lib::arrayant_combine_pattern_multi(
    const std::vector<quadriga_lib::arrayant<float>> &,
    const arma::Col<float> *, const arma::Col<float> *, const arma::Col<float> *);

template std::vector<quadriga_lib::arrayant<double>> quadriga_lib::arrayant_combine_pattern_multi(
    const std::vector<quadriga_lib::arrayant<double>> &,
    const arma::Col<double> *, const arma::Col<double> *, const arma::Col<double> *);