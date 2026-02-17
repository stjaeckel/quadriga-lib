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
