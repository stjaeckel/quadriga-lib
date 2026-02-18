// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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

#include "quadriga_arrayant.hpp"
#include "qd_arrayant_functions.hpp"

#include <stdexcept>
#include <filesystem>

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# qdant_write_multi
Write a vector of array antenna objects to a single QDANT file

## Description:
- Writes multiple `arrayant` objects to a single QuaDRiGa array antenna exchange format (QDANT) file,
  one per sequential ID.
- Each entry in the input vector is assigned a 1-based ID (1, 2, ..., n_entries) and written to the file
  using the `arrayant::qdant_write` member function.
- A layout matrix of size `[n_entries, 1]` is created automatically with entries `1, 2, ..., n_entries`
  to describe the organization inside the file.
- If the output file already exists, it is deleted before writing to ensure a clean file.
- All arrayant objects in the vector are validated before writing. An exception is thrown if any entry is invalid.
- This function is the primary mechanism for storing frequency-dependent models, where each arrayant
  in the vector represents the directivity pattern at a specific frequency. The corresponding frequency
  is stored in the `center_frequency` field of each arrayant.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::qdant_write_multi(
        const std::string &fn,
        const std::vector<arrayant<dtype>> &arrayant_vec)
```

## Arguments:
- `const std::string &**fn**` (input)<br>
  Filename of the QDANT file to write. If the file already exists, it is overwritten. Cannot be empty.

- `const std::vector<arrayant<dtype>> &**arrayant_vec**` (input)<br>
  Vector of arrayant objects to write. Each entry is stored with a sequential 1-based ID. The vector must not be empty and all entries must be valid arrayant objects.

## Returns:
- `void`

## Example:
```
// Generate a loudspeaker model and write to file
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
                12.0, 12.0, 85.0, "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);

quadriga_lib::qdant_write_multi("speaker.qdant", spk);

// Read back a specific frequency entry (e.g. the 3rd one, ID = 3)
auto ant = quadriga_lib::qdant_read<double>("speaker.qdant", 3);
```

## See also:
- <a href="#arrayant">arrayant</a>
- <a href="#.qdant_write">arrayant.qdant_write</a>
- <a href="#qdant_read">qdant_read</a>
- <a href="#generate_speaker">generate_speaker</a>
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
Reads array antenna data from QDANT files

## Description:
- Reads antenna pattern data and layout from a QuaDRiGa array antenna exchange format (QDANT) file,
  which stores antenna pattern data in XML format.
- Returns an antenna array (`arrayant`) object constructed from the specified data in the QDANT file.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
arrayant<dtype> quadriga_lib::qdant_read(std::string fn, unsigned id = 1, arma::u32_mat *layout = nullptr);
```

## Arguments:
- `std::string **fn**` (input)<br>
  Filename of the QDANT file from which antenna pattern data will be read. Cannot be empty.

- `unsigned **id** = 1` (optional input)<br>
  ID of the antenna within the QDANT file to read. Default is `1`.

- `arma::u32_mat **layout** = nullptr` (optional output)<br>
  Pointer to a matrix that will store the layout information of multiple antenna elements from the file. The layout contains element IDs present in the QDANT file.

## Returns:
- `arrayant<dtype>`<br>
  Antenna array object containing data from the specified antenna ID within the QDANT file.

## Example:
```
arma::u32_mat layout;
auto ant = quadriga_lib::qdant_read<double>("antenna_data.qdant", 2, &layout);
```

## See also:
- <a href="#arrayant">arrayant</a>
- <a href="#.qdant_write">arrayant.qdant_write</a>
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#6cab4884">QDANT</a>)
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
Read all array antenna objects from a QDANT file

## Description:
- Reads all `arrayant` objects stored in a QuaDRiGa array antenna exchange format (QDANT) file and
  returns them as a vector.
- The file layout is obtained by probing the file with ID 1. Even if ID 1 does not exist as an entry,
  the internal reader still returns the layout describing the file contents. This layout is then used
  to determine which IDs to read.
- Unique non-zero IDs are extracted from the layout in the order of their first appearance (scanning rows
  within each column). Each unique ID is read exactly once, regardless of how many times it appears in the layout.
- All read arrayant objects are validated. An exception is thrown if any entry cannot be read or fails validation.
- This function is the counterpart to `qdant_write_multi` and the primary mechanism for loading frequency-dependent
  models. The frequency corresponding to each entry is stored in the `center_frequency` field of the returned arrayant objects.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
std::vector<arrayant<dtype>> quadriga_lib::qdant_read_multi(
        const std::string &fn,
        arma::u32_mat *layout = nullptr)
```

## Arguments:
- `const std::string &**fn**` (input)<br>
  Filename of the QDANT file to read. Cannot be empty.

- `arma::u32_mat ***layout** = nullptr` (optional output)<br>
  Pointer to a matrix that will receive the layout information from the file. The layout describes the organization of array antenna entries inside the file, with non-zero values corresponding to entry IDs. If `nullptr`, the layout is not returned.

## Returns:
- `std::vector<arrayant<dtype>>`<br>
  Vector of arrayant objects, one per unique ID found in the layout. Entries are ordered by their first appearance in the layout. Each arrayant contains the full pattern data including `center_frequency`.

## Example:
```
// Read all entries from a file
auto spk = quadriga_lib::qdant_read_multi<double>("speaker.qdant");

// Read with layout information
arma::u32_mat layout;
auto spk2 = quadriga_lib::qdant_read_multi<double>("speaker.qdant", &layout);
std::cout << "File contains " << spk2.size() << " entries" << std::endl;
std::cout << "Layout: " << std::endl;
layout.print();
```

## See also:
- <a href="#arrayant">arrayant</a>
- <a href="#qdant_read">qdant_read</a>
- <a href="#qdant_write_multi">qdant_write_multi</a>
- <a href="#generate_speaker">generate_speaker</a>
MD!*/

template <typename dtype>
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read_multi(const std::string &fn, arma::u32_mat *layout)
{
    if (fn.empty())
        throw std::invalid_argument("qdant_read_multi: Filename cannot be empty.");

    // --- Step 1: Probe the file with id=1 to obtain the layout ---
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

    // --- Step 2: Extract unique IDs from the layout in order of first appearance ---
    // Scan row-major (column by column in Armadillo's column-major storage → row by row logically)
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

    // --- Step 3: Read each unique ID ---
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

    // --- Step 4: Optionally return the layout ---
    if (layout != nullptr)
        *layout = file_layout;

    return result;
}

template std::vector<quadriga_lib::arrayant<float>> quadriga_lib::qdant_read_multi(const std::string &, arma::u32_mat *);
template std::vector<quadriga_lib::arrayant<double>> quadriga_lib::qdant_read_multi(const std::string &, arma::u32_mat *);

/*!MD
# arrayant_is_valid_multi
Validate a vector of array antenna objects for multi-frequency consistency

## Description:
- Validates that a vector of `arrayant` objects forms a consistent multi-frequency model
- Each individual entry is validated using its `is_valid` member function. The `quick_check` parameter
  is passed through to control whether pointer-based fast validation or full deep validation is performed per entry.
- Beyond individual validity, the function checks that all entries in the vector are structurally compatible by verifying:
- Azimuth and elevation grids have identical sizes and values across all entries.
- The number of antenna elements is the same for all entries.
- Element positions are identical across all entries (physical antenna/driver layout does not change with frequency).
- Coupling matrices have the same shape (same number of elements and ports) across all entries.
- The function does not check pattern data values, center frequencies, or coupling matrix values, as these are expected to vary with frequency.
- Validation stops at the first error found and returns a descriptive message indicating which entry and what property failed.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
std::string quadriga_lib::arrayant_is_valid_multi(
        const std::vector<arrayant<dtype>> &arrayant_vec,
        bool quick_check = true)
```

## Arguments:
- `const std::vector<arrayant<dtype>> &**arrayant_vec**` (input)<br>
  Vector of arrayant objects to validate. Must not be empty.

- `bool **quick_check** = true` (optional input)<br>
  If `true`, individual entry validation uses the fast pointer-based check (skips deep validation when `check_ptr` matches). If `false`, performs a full deep validation of every entry. Default: `true`.

## Returns:
- `std::string`<br>
  Empty string if the vector is valid and all entries are consistent. Otherwise, a descriptive error message indicating which entry (by index) and what property caused the validation failure. For example: `"Entry 3: Azimuth grid values do not match entry 0."`.

## Example:
```
// Generate a speaker model and validate
auto spk = quadriga_lib::generate_speaker<double>("piston");
std::string err = quadriga_lib::is_valid_multi(spk, false);
if (!err.empty())
    std::cerr << "Validation error: " << err << std::endl;

// Detect tampered element positions
spk[2].element_pos(0, 0) = 999.0;
err = quadriga_lib::is_valid_multi(spk, false);
// Returns: "Entry 2: Element positions do not match entry 0."
```

## See also:
- <a href="#arrayant">arrayant</a>
- <a href="#.is_valid">arrayant.is_valid</a>
- <a href="#generate_speaker">generate_speaker</a>
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
Copy antenna elements across all entries in a multi-frequency arrayant vector

## Description:
- Copies the pattern data, element position, and coupling weights from a source element to one or more
  destination elements, applied identically to every frequency entry in the vector.
- Calls the `arrayant::copy_element` member function for each entry in the vector. This member function handles
  all internal details including pattern data copying, element position copying, and coupling matrix management.
- If any destination index exceeds the current number of elements, the arrayant is automatically enlarged.
  New elements receive an identity coupling entry (one additional port per added element), consistent with
  the behavior of the underlying `arrayant::copy_element` member function.
- Source and destination indices are 0-based.
- Two overloads are provided: one accepting a vector of destination indices (`arma::uvec`) for copying
  to multiple targets, and one accepting a single destination index (`arma::uword`).
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant_copy_element_multi(
        std::vector<arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uvec destination)

void quadriga_lib::arrayant_copy_element_multi(
        std::vector<arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uword destination)
```

## Arguments:
- `std::vector<arrayant<dtype>> &**arrayant_vec**` (input/output)<br>
  Vector of arrayant objects to update. Modified in-place. Must not be empty. All entries must be valid arrayant objects.

- `arma::uword **source**` (input)<br>
  0-based index of the source element to copy from. Must be within the current element count.

- `arma::uvec **destination**` (input)<br>
  0-based indices of the destination elements to copy to. If any index exceeds the current element count, the arrayant is enlarged accordingly. Must not be empty.

- `arma::uword **destination**` (input)<br>
  Single 0-based destination element index (convenience overload).

## Returns:
- `void`

## Example:
```
// Build a 4-element line array from a single driver template
arma::vec freqs = {500.0, 1000.0, 2000.0, 5000.0};
auto driver = quadriga_lib::generate_speaker<double>(
    "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
    0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

// Replicate element 0 to create elements 1, 2, 3
arma::uvec dest = {1, 2, 3};
quadriga_lib::arrayant_copy_element_multi(driver, 0, dest);

// Now set the individual positions (vertical line array, 15 cm spacing)
arma::mat positions(3, 4, arma::fill::zeros);
positions(2, 0) = -0.225;  // z = -22.5 cm
positions(2, 1) = -0.075;  // z = -7.5 cm
positions(2, 2) =  0.075;  // z = +7.5 cm
positions(2, 3) =  0.225;  // z = +22.5 cm
quadriga_lib::arrayant_set_element_pos_multi(driver, positions);

// Result: 4 identical elements, 4 independent ports, per frequency entry
```

## See also:
- <a href="#.copy_element">arrayant.copy_element</a>
- <a href="#arrayant_set_element_pos_multi">arrayant_set_element_pos_multi</a>
- <a href="#arrayant_concat_multi">arrayant_concat_multi</a>
- <a href="#generate_speaker">generate_speaker</a>
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

## Description:
- Updates the `element_pos` field of every `arrayant` in a multi-frequency vector in-place.
- If `i_element` is empty (default), all element positions are replaced. In this case, `element_pos`
  must have exactly `n_elements` columns matching the number of elements in the arrayant.
- If `i_element` is provided, only the specified elements (0-based indices) are updated. The number
  of columns in `element_pos` must match the number of indices in `i_element`.
- All entries in the vector must have the same number of elements. If any entry has a different
  element count, an exception is thrown.
- If an entry's `element_pos` is empty (uninitialized), it is initialized to zeros before
  applying the update.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant_set_element_pos_multi(
        std::vector<arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        arma::uvec i_element = arma::uvec())
```

## Arguments:
- `std::vector<arrayant<dtype>> &**arrayant_vec**` (input/output)<br>
  Vector of arrayant objects to update. Modified in-place. Must not be empty.

- `const arma::Mat<dtype> &**element_pos**` (input)<br>
  New element positions with size `[3, n_update]`, where each column contains the (x, y, z) coordinates of one element in meters.

- `arma::uvec **i_element** = arma::uvec()` (optional input)<br>
  0-based indices of the elements to update. If empty, all elements are updated and `element_pos` must have `n_elements` columns. If provided, `element_pos` must have the same number of columns as `i_element` has entries.

## Example:
```
// Generate a two-driver speaker (woofer + tweeter) at multiple frequencies
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0};
auto woofer = quadriga_lib::generate_speaker<double>("piston", 0.08, 50.0, 2000.0);
auto tweeter = quadriga_lib::generate_speaker<double>("horn", 0.025, 1500.0, 20000.0);

// Set the woofer position: centered, 10 cm below the tweeter
arma::mat pos = arma::mat({0.0, 0.0, -0.10}).t();
quadriga_lib::arrayant_set_element_pos_multi(woofer, pos);

// Or update a specific element in a multi-element arrayant
arma::mat tweeter_pos = arma::mat({0.0, 0.0, 0.10}).t();
arma::uvec idx = {0};
quadriga_lib::arrayant_set_element_pos_multi(tweeter, tweeter_pos, idx);
```

## See also:
- <a href="#arrayant">arrayant</a>
- <a href="#generate_speaker">generate_speaker</a>
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
Rotate antenna patterns across all entries in a multi-frequency arrayant vector

## Description:
- Applies Euler rotations to every frequency entry in a multi-frequency `arrayant` vector,
  modifying the patterns in-place.
- Grid adjustment is always disabled because multi-frequency loudspeaker models use uniform angular
  grids that must remain consistent across all frequency entries. Non-uniform grid resampling would break this consistency.
- If `i_element` is empty (default), all elements are rotated. If `i_element` is provided, only the specified
  elements (0-based indices) are rotated, leaving others unchanged. This is useful for rotating individual drivers in a multi-driver speaker model.
- For acoustic speaker models where the pressure field is scalar (stored only in `e_theta_re`), usage mode 1
  (pattern only) is recommended to avoid spurious polarization effects.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant_rotate_pattern_multi(
        std::vector<arrayant<dtype>> &arrayant_vec,
        dtype x_deg = 0.0,
        dtype y_deg = 0.0,
        dtype z_deg = 0.0,
        unsigned usage = 0,
        arma::uvec i_element = arma::uvec())
```

## Arguments:
- `std::vector<arrayant<dtype>> &**arrayant_vec**` (input/output)<br>
  Vector of arrayant objects to update. Modified in-place. Must not be empty.

- `dtype **x_deg** = 0.0` (optional input)<br>
  Rotation angle around the x-axis (bank angle) in degrees. Default: `0.0`.

- `dtype **y_deg** = 0.0` (optional input)<br>
  Rotation angle around the y-axis (tilt angle) in degrees. Default: `0.0`.

- `dtype **z_deg** = 0.0` (optional input)<br>
  Rotation angle around the z-axis (heading angle) in degrees. Default: `0.0`.

- `unsigned **usage** = 0` (optional input)<br>
  Rotation mode: `0` = rotate both pattern and polarization, `1` = rotate only pattern, `2` = rotate only polarization. Grid adjustment is always disabled.

- `arma::uvec **i_element** = arma::uvec()` (optional input)<br>
  0-based indices of the elements to rotate. If empty, all elements are rotated. If provided, only the specified elements are affected.

## Usage Mode Mapping:

 Input Mode | Behavior                   | Internal rotate_pattern Mode
------------|----------------------------|-----------------------------
          0 | Pattern + polarization     | 3 (no grid adjustment)
          1 | Pattern only               | 4 (no grid adjustment)
          2 | Polarization only          | 2 (no grid adjustment)

## Example:
```
// Generate a 2-way speaker model
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto woofer = quadriga_lib::generate_speaker<double>(
    "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
    0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
auto tweeter = quadriga_lib::generate_speaker<double>(
    "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
    0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);

// Rotate entire speaker 30 degrees around the z-axis (heading)
quadriga_lib::arrayant_rotate_pattern_multi(speaker, 0.0, 0.0, 30.0, 1);

// Tilt only the tweeter (element 1) upward by 10 degrees
arma::uvec tweeter_idx = {1};
quadriga_lib::arrayant_rotate_pattern_multi(speaker, 0.0, 10.0, 0.0, 1, tweeter_idx);
```

## See also:
- <a href="#.rotate_pattern">arrayant.rotate_pattern</a>
- <a href="#arrayant_concat_multi">arrayant_concat_multi</a>
- <a href="#arrayant_set_element_pos_multi">arrayant_set_element_pos_multi</a>
- <a href="#generate_speaker">generate_speaker</a>
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

## Description:
- Combines two multi-frequency `arrayant` vectors into a single vector representing a multi-driver
  loudspeaker (or any multi-element antenna array).
- Both input vectors must have the same number of frequency entries, identical angular grids (azimuth and elevation),
  and matching center frequencies at each index.
- For each frequency entry, the function performs:
- **Pattern concatenation**: The 3D pattern cubes (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`)
  are joined along the element (slice) dimension, producing `n_elem1 + n_elem2` elements.
- **Element position concatenation**: The `element_pos` matrices are horizontally concatenated.
  If either input has empty positions, zeros are substituted.
- **Block-diagonal coupling assembly**: The coupling matrices are assembled in block-diagonal form,
  preserving independent port sets from each input. If vec1 has `p1` ports and vec2 has `p2` ports, the
  output has `p1 + p2` ports with no cross-coupling between the two driver groups.
- Both inputs are validated with `is_valid_multi` before processing, and each output entry is
  validated with `validate` before returning.
- The output inherits the name, azimuth grid, elevation grid, and center frequency from the first
  input vector.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
std::vector<arrayant<dtype>> quadriga_lib::arrayant_concat_multi(
        const std::vector<arrayant<dtype>> &arrayant_vec1,
        const std::vector<arrayant<dtype>> &arrayant_vec2)
```

## Arguments:
- `const std::vector<arrayant<dtype>> &**arrayant_vec1**` (input)<br>
  First arrayant vector. All entries must be valid and mutually consistent (same grids, element counts, coupling shape).

- `const std::vector<arrayant<dtype>> &**arrayant_vec2**` (input)<br>
  Second arrayant vector. Must have the same number of entries as `arrayant_vec1`, with matching angular grids and center frequencies.

## Returns:
- `std::vector<arrayant<dtype>>`<br>
  Combined arrayant vector with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports per frequency entry. The coupling matrix has block-diagonal structure.

## Coupling Matrix Structure:
For a woofer with 1 element / 1 port and a tweeter with 1 element / 1 port, the combined coupling matrix is:
```
                    Port 1 (woofer)   Port 2 (tweeter)
Elem 1 (woofer)     1.0                0.0
Elem 2 (tweeter)    0.0                1.0
```

For more complex arrays (e.g. a line array with 4 elements / 2 ports combined with a tweeter of 1 element / 1 port):
```
     P1    P2    P3
E1   c11   c12   0
E2   c21   c22   0
E3   c31   c32   0
E4   c41   c42   0
E5   0     0     1
```

## Example:
```
// Build a 2-way loudspeaker: woofer + tweeter at matching frequencies
arma::vec freqs = {100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0};

// Generate woofer: 6.5" piston, sealed box, 50-3000 Hz passband
auto woofer = quadriga_lib::generate_speaker<double>(
    "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
    0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);

// Generate tweeter: 1" dome, sealed box, 1500-20000 Hz passband
auto tweeter = quadriga_lib::generate_speaker<double>(
    "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
    0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);

// Position the drivers: woofer centered, tweeter 12 cm above
arma::mat woofer_pos = arma::mat({0.0, 0.0, 0.0}).t();
arma::mat tweeter_pos = arma::mat({0.0, 0.0, 0.12}).t();
quadriga_lib::arrayant_set_element_pos_multi(woofer, woofer_pos);
quadriga_lib::arrayant_set_element_pos_multi(tweeter, tweeter_pos);

// Combine into a single 2-way speaker model
auto speaker_2way = quadriga_lib::arrayant_concat_multi(woofer, tweeter);

// Result: 2 elements, 2 ports per frequency entry
// Port 1 drives the woofer, Port 2 drives the tweeter
// Crossover behavior emerges from overlapping bandpass responses

// Write to file
quadriga_lib::qdant_write_multi("speaker_2way.qdant", speaker_2way);
```

## See also:
- <a href="#arrayant">arrayant</a>
- <a href="#generate_speaker">generate_speaker</a>
- <a href="#arrayant_set_element_pos_multi">arrayant_set_element_pos_multi</a>
- <a href="#arrayant_is_valid_multi">arrayant_is_valid_multi</a>
- <a href="#qdant_write_multi">qdant_write_multi</a>
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
