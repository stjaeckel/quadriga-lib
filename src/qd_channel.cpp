// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"

#include <iomanip> // std::setprecision

/*!SECTION
Channel class
SECTION!*/

/*!MD
# channel<++>
Class for storing and managing MIMO channel data and metadata across multiple snapshots

## Description:
- Represents path-level MIMO channel data between antenna arrays over multiple time snapshots
- Each snapshot may have a different number of propagation paths `n_path`
- Unstructured metadata supported via `par_names` / `par_data`
- Allowed datatypes: `float` or `double`

## Attributes:

| Attribute | Size | Description |
|-----------|------|-------------|
| `std::string name` | — | Name of the channel object |
| `arma::Col<dtype> center_frequency` | `[1]`, `[n_snap]`, or `[]` | Center frequency in Hz |
| `arma::Mat<dtype> tx_pos` | `[3, n_snap]` or `[3, 1]` | Transmitter positions |
| `arma::Mat<dtype> rx_pos` | `[3, n_snap]` or `[3, 1]` | Receiver positions |
| `arma::Mat<dtype> tx_orientation` | `[3, n_snap]`, `[3, 1]`, or `[]` | Transmitter orientation (Euler angles) |
| `arma::Mat<dtype> rx_orientation` | `[3, n_snap]`, `[3, 1]`, or `[]` | Receiver orientation (Euler angles) |
| `std::vector<arma::Cube<dtype>> coeff_re` | `[n_rx, n_tx, n_path]` per snap | Channel coefficients, real part |
| `std::vector<arma::Cube<dtype>> coeff_im` | `[n_rx, n_tx, n_path]` per snap | Channel coefficients, imaginary part |
| `std::vector<arma::Cube<dtype>> delay` | `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` per snap | Path delays in seconds |
| `std::vector<arma::Col<dtype>> path_gain` | `[n_path]` per snap | Path gains before antenna pattern |
| `std::vector<arma::Col<dtype>> path_length` | `[n_path]` per snap | Path lengths TX to RX in meters |
| `std::vector<arma::Mat<dtype>> path_polarization` | `[8, n_path]` per snap | Interleaved polarization transfer matrices |
| `std::vector<arma::Mat<dtype>> path_angles` | `[n_path, 4]` per snap | Angles {AOD, EOD, AOA, EOA} in rad |
| `std::vector<arma::Mat<dtype>> path_fbs_pos` | `[3, n_path]` per snap | First-bounce scatterer positions |
| `std::vector<arma::Mat<dtype>> path_lbs_pos` | `[3, n_path]` per snap | Last-bounce scatterer positions |
| `std::vector<arma::Col<unsigned>> no_interact` | `[n_path]` per snap | Number of interactions per path |
| `std::vector<arma::Mat<dtype>> interact_coord` | `[3, sum(no_interact)]` per snap | Interaction point coordinates |
| `std::vector<std::string> par_names` | — | Names of unstructured metadata fields |
| `std::vector<std::any> par_data` | — | Unstructured metadata values (string, scalar, matrix, etc.) |
| `int initial_position` | scalar | 0-based index of the reference snapshot |

## Simple member functions:

| Method | Description |
|---|---|
| `.n_snap()` | Returns the number of snapshots |
| `.n_rx()` | Returns number of receive antennas; 0 if coefficients absent |
| `.n_tx()` | Returns number of transmit antennas; 0 if coefficients absent |
| `.n_path()` | Returns number of paths per snapshot as a vector |
| `.empty()` | Returns true if the object contains no channel data |
| `.is_valid()` | Returns empty string if valid, otherwise an error message |

## Complex member functions:
- [[.add_paths]]
- [[.calc_effective_path_gain]]
- [[.write_paths_to_obj_file]]
MD!*/

// CHANNEL METHODS : Return object dimensions
template <typename dtype>
arma::uword quadriga_lib::channel<dtype>::n_snap() const
{
    arma::uword s_arma = 0ULL;

    if (center_frequency.n_elem > s_arma) // 1 or s
        s_arma = center_frequency.n_elem;

    if (tx_orientation.n_cols > s_arma) // 1 or s
        s_arma = tx_orientation.n_cols;

    if (rx_orientation.n_cols > s_arma) // 1 or s
        s_arma = rx_orientation.n_cols;

    if (rx_pos.n_cols > s_arma) // 1 or s
        s_arma = rx_pos.n_cols;

    if (tx_pos.n_cols > s_arma) // 1 or s
        s_arma = tx_pos.n_cols;

    size_t s_vector = (size_t)s_arma;

    if (coeff_re.size() > s_vector)
        s_vector = coeff_re.size();

    if (delay.size() > s_vector)
        s_vector = delay.size();

    if (path_gain.size() > s_vector)
        s_vector = path_gain.size();

    if (path_length.size() > s_vector)
        s_vector = path_length.size();

    if (path_polarization.size() > s_vector)
        s_vector = path_polarization.size();

    if (path_angles.size() > s_vector)
        s_vector = path_angles.size();

    if (path_fbs_pos.size() > s_vector)
        s_vector = path_fbs_pos.size();

    if (path_lbs_pos.size() > s_vector)
        s_vector = path_lbs_pos.size();

    if (no_interact.size() > s_vector)
        s_vector = no_interact.size();

    return (arma::uword)s_vector;
}

template <typename dtype>
arma::uword quadriga_lib::channel<dtype>::n_rx() const
{
    if (coeff_re.size() != 0)
        return coeff_re[0].n_rows;
    else
        return 0;
}

template <typename dtype>
arma::uword quadriga_lib::channel<dtype>::n_tx() const
{
    if (coeff_re.size() != 0)
        return coeff_re[0].n_cols;
    else
        return 0;
}

template <typename dtype>
arma::uvec quadriga_lib::channel<dtype>::n_path() const
{
    size_t n_snap = (size_t)this->n_snap();
    if (n_snap == 0)
        return arma::uvec();

    arma::uvec n_path(n_snap);
    auto *p_path = n_path.memptr();
    if (coeff_re.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = coeff_re[i].n_slices;
    else if (no_interact.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = no_interact[i].n_elem;
    else if (path_gain.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_gain[i].n_elem;
    else if (path_length.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_length[i].n_elem;
    else if (path_polarization.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_polarization[i].n_cols;
    else if (path_angles.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_angles[i].n_rows;
    else if (path_fbs_pos.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_fbs_pos[i].n_cols;
    else if (path_lbs_pos.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_lbs_pos[i].n_cols;

    return n_path;
}

// Returns true if the channel object contains no data
template <typename dtype>
bool quadriga_lib::channel<dtype>::empty() const
{
    if (center_frequency.n_elem != 0ULL)
        return false;

    if (tx_pos.n_elem != 0ULL)
        return false;

    if (rx_pos.n_elem != 0ULL)
        return false;

    if (tx_orientation.n_elem != 0ULL)
        return false;

    if (rx_orientation.n_elem != 0ULL)
        return false;

    if (coeff_re.size() != 0)
        return false;

    if (delay.size() != 0)
        return false;

    if (path_gain.size() != 0)
        return false;

    if (path_length.size() != 0)
        return false;

    if (path_polarization.size() != 0)
        return false;

    if (path_angles.size() != 0)
        return false;

    if (path_fbs_pos.size() != 0)
        return false;

    if (path_lbs_pos.size() != 0)
        return false;

    if (no_interact.size() != 0)
        return false;

    if (interact_coord.size() != 0)
        return false;

    if (initial_position != 0)
        return false;

    return true;
}

// CHANNEL METHOD : Validate correctness of the members
template <typename dtype>
std::string quadriga_lib::channel<dtype>::is_valid() const
{
    arma::uword n_snap_arma = this->n_snap();
    size_t n_snap_vector = (size_t)n_snap_arma;
    arma::uword n_tx = 0ULL, n_rx = 0ULL;
    arma::uvec n_pth_v = this->n_path();

    if (name.length() > 255)
        return std::string("Name can have at most 255 characters.");

    if (n_pth_v.n_elem != n_snap_arma)
        return std::string("Number of elements returned by 'n_path()' does not match number of snapshots.");
    auto n_pth = n_pth_v.memptr();

    if (n_snap_arma != 0ULL && rx_pos.n_rows != 3ULL)
        return std::string("'rx_pos' is missing or ill-formatted (must have 3 rows).");

    if (rx_pos.n_cols != 1ULL && rx_pos.n_cols != n_snap_arma)
        return std::string("Number of columns in 'rx_pos' must be 1 or match the number of snapshots.");

    if (n_snap_arma != 0ULL && tx_pos.n_rows != 3ULL)
        return std::string("'tx_pos' is missing or ill-formatted (must have 3 rows).");

    if (tx_pos.n_cols != 1ULL && tx_pos.n_cols != n_snap_arma)
        return std::string("Number of columns in 'tx_pos' must be 1 or match the number of snapshots.");

    if (center_frequency.n_elem != 0ULL && center_frequency.n_elem != 1ULL && center_frequency.n_elem != n_snap_arma)
        return std::string("Number of entries in 'center_frequency' must be 0, 1 or match the number of snapshots.");

    if (tx_orientation.n_elem != 0ULL && tx_orientation.n_rows != 3ULL)
        return std::string("'tx_orientation' must be empty or have 3 rows.");

    if (tx_orientation.n_elem != 0ULL && tx_orientation.n_cols != 1ULL && tx_orientation.n_cols != n_snap_arma)
        return std::string("Number of columns in 'tx_orientation' must be 1 or match the number of snapshots.");

    if (rx_orientation.n_elem != 0ULL && rx_orientation.n_rows != 3ULL)
        return std::string("'rx_orientation' must be empty or have 3 rows.");

    if (rx_orientation.n_elem != 0ULL && rx_orientation.n_cols != 1ULL && rx_orientation.n_cols != n_snap_arma)
        return std::string("Number of columns in 'rx_orientation' must be 1 or match the number of snapshots.");

    if (coeff_re.size() != 0 && coeff_re.size() != n_snap_vector)
        return std::string("'coeff_re' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && coeff_re.size() == n_snap_vector)
    {
        if (coeff_im.size() != n_snap_vector)
            return std::string("Imaginary part of channel coefficients 'coeff_im' is missing or incomplete.");

        if (delay.size() != n_snap_vector)
            return std::string("Delays are missing or incomplete.");

        n_rx = coeff_re[0].n_rows, n_tx = coeff_re[0].n_cols;

        for (size_t i = 0; i < n_snap_vector; ++i)
        {
            if (coeff_re[i].n_rows != n_rx || coeff_re[i].n_cols != n_tx || coeff_re[i].n_slices != n_pth[i])
                return std::string("Size mismatch in 'coeff_re[" + std::to_string(i) + "]'.");

            if (coeff_im[i].n_rows != n_rx || coeff_im[i].n_cols != n_tx || coeff_im[i].n_slices != n_pth[i])
                return std::string("Size mismatch in 'coeff_im[" + std::to_string(i) + "]'.");

            if (!(delay[i].n_rows == n_rx && delay[i].n_cols == n_tx && delay[i].n_slices == n_pth[i]) &&
                !(delay[i].n_rows == 1 && delay[i].n_cols == 1 && delay[i].n_slices == n_pth[i]))
                return std::string("Size mismatch in 'delay[" + std::to_string(i) + "]'.");
        }
    }
    else if (!coeff_im.empty())
        return std::string("Real part of channel coefficients 'coeff_re' is missing or incomplete.");
    else if (!delay.empty())
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (delay[i].n_rows != 1 || delay[i].n_cols != 1 || delay[i].n_slices != n_pth[i])
                return std::string("Size mismatch in 'delay[" + std::to_string(i) + "]'.");

    if (path_gain.size() != 0 && path_gain.size() != n_snap_vector)
        return std::string("'path_gain' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_gain.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_gain[i].n_elem != n_pth[i])
                return std::string("Size mismatch in 'path_gain[" + std::to_string(i) + "]'.");

    if (path_length.size() != 0 && path_length.size() != n_snap_vector)
        return std::string("'path_length' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_length.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_length[i].n_elem != n_pth[i])
                return std::string("Size mismatch in 'path_length[" + std::to_string(i) + "]'.");

    if (path_polarization.size() != 0 && path_polarization.size() != n_snap_vector)
        return std::string("'path_polarization' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_polarization.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_polarization[i].n_rows != 8ULL || path_polarization[i].n_cols != n_pth[i])
                return std::string("Size mismatch in 'path_polarization[" + std::to_string(i) + "]'.");

    if (path_angles.size() != 0 && path_angles.size() != n_snap_vector)
        return std::string("'path_angles' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_angles.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_angles[i].n_rows != n_pth[i] || path_angles[i].n_cols != 4ULL)
                return std::string("Size mismatch in 'path_angles[" + std::to_string(i) + "]'.");

    if (path_fbs_pos.size() != 0 && path_fbs_pos.size() != n_snap_vector)
        return std::string("'path_fbs_pos' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_fbs_pos.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_fbs_pos[i].n_rows != 3ULL || path_fbs_pos[i].n_cols != n_pth[i])
                return std::string("Size mismatch in 'path_fbs_pos[" + std::to_string(i) + "]'.");

    if (path_lbs_pos.size() != 0 && path_lbs_pos.size() != n_snap_vector)
        return std::string("'path_lbs_pos' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_lbs_pos.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_lbs_pos[i].n_rows != 3ULL || path_lbs_pos[i].n_cols != n_pth[i])
                return std::string("Size mismatch in 'path_lbs_pos[" + std::to_string(i) + "]'.");

    if (no_interact.size() != 0 && no_interact.size() != n_snap_vector)
        return std::string("'no_interact' must be empty or match the number of snapshots.");
    else if (no_interact.size() == n_snap_vector && interact_coord.size() != n_snap_vector)
        return std::string("'no_interact' is provided but 'interact_coord' is missing or has wrong number of snapshots.");
    else if (no_interact.size() == 0 && interact_coord.size() != 0)
        return std::string("'interact_coord' is provided but 'no_interact' is missing.");
    else if (no_interact.size() == n_snap_vector && interact_coord.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
        {
            if (no_interact[i].n_elem != n_pth[i])
                return std::string("Size mismatch in 'no_interact[" + std::to_string(i) + "]'.");

            if (interact_coord[i].n_rows != 3ULL)
                return std::string("'interact_coord[" + std::to_string(i) + "]' must have 3 rows.");

            unsigned cnt = 0;
            for (const unsigned *p = no_interact[i].begin(); p < no_interact[i].end(); ++p)
                cnt += *p;

            if (interact_coord[i].n_cols != (arma::uword)cnt)
                return std::string("Number of columns in 'interact_coord[" + std::to_string(i) + "]' must match the sum of 'no_interact'.");
        }

    if (par_names.size() != par_data.size())
        return std::string("Number of elements in 'par_data' must match number of elements in 'par_name'.");

    for (size_t i = 0; i < par_names.size(); ++i)
        if (any_type_id(&par_data[i]) < 0)
            return std::string("Unsupported datatype in unstructured data.");

    return std::string("");
}

/*!MD
# .add_paths
Append new propagation paths to an existing channel snapshot

## Description:
- Adds path-level data to snapshot `i_snap` in a `channel` object; does not modify `tx_pos`, `rx_pos`, or orientation fields
- All provided fields must have consistent length `n_path_add` and match existing snapshot structure
- Member function of [[channel]]
- Allowed datatypes: `float` or `double`

## Declaration:
```
void quadriga_lib::channel<dtype>::add_paths(
    arma::uword i_snap,
    const arma::Cube<dtype> *coeff_re_add = nullptr,
    const arma::Cube<dtype> *coeff_im_add = nullptr,
    const arma::Cube<dtype> *delay_add = nullptr,
    const arma::u32_vec *no_interact_add = nullptr,
    const arma::Mat<dtype> *interact_coord_add = nullptr,
    const arma::Col<dtype> *path_gain_add = nullptr,
    const arma::Col<dtype> *path_length_add = nullptr,
    const arma::Mat<dtype> *path_polarization_add = nullptr,
    const arma::Mat<dtype> *path_angles_add = nullptr,
    const arma::Mat<dtype> *path_fbs_pos_add = nullptr,
    const arma::Mat<dtype> *path_lbs_pos_add = nullptr);
```

## Input Arguments:
- **`i_snap`** — 0-based snapshot index to append paths to
- **`coeff_re_add`** *(optional)* — Real part of channel coefficients, `[n_rx, n_tx, n_path_add]`
- **`coeff_im_add`** *(optional)* — Imaginary part of channel coefficients, `[n_rx, n_tx, n_path_add]`
- **`delay_add`** *(optional)* — Propagation delays in seconds, `[n_rx, n_tx, n_path_add]` or `[1, 1, n_path_add]`
- **`no_interact_add`** *(optional)* — Number of interaction points per path, `[n_path_add]`
- **`interact_coord_add`** *(optional)* — Interaction point coordinates, `[3, sum(no_interact)]`
- **`path_gain_add`** *(optional)* — Path gains before antenna effects, `[n_path_add]`
- **`path_length_add`** *(optional)* — Path lengths from TX to RX phase center in meters, `[n_path_add]`
- **`path_polarization_add`** *(optional)* — Interleaved polarization transfer matrices, `[8, n_path_add]`
- **`path_angles_add`** *(optional)* — Departure/arrival angles {AOD, EOD, AOA, EOA} in rad, `[n_path_add, 4]`
- **`path_fbs_pos_add`** *(optional)* — First-bounce scatterer positions, `[3, n_path_add]`
- **`path_lbs_pos_add`** *(optional)* — Last-bounce scatterer positions, `[3, n_path_add]`
MD!*/

template <typename dtype>
void quadriga_lib::channel<dtype>::add_paths(arma::uword i_snap,
                                             const arma::Cube<dtype> *coeff_re_add, const arma::Cube<dtype> *coeff_im_add, const arma::Cube<dtype> *delay_add,
                                             const arma::u32_vec *no_interact_add, const arma::Mat<dtype> *interact_coord_add,
                                             const arma::Col<dtype> *path_gain_add, const arma::Col<dtype> *path_length_add,
                                             const arma::Mat<dtype> *path_polarization_add, const arma::Mat<dtype> *path_angles_add,
                                             const arma::Mat<dtype> *path_fbs_pos_add, const arma::Mat<dtype> *path_lbs_pos_add)
{
    std::string error_message = this->is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    arma::uword n_snap_chan = this->n_snap();
    if (i_snap >= n_snap_chan)
        throw std::invalid_argument("Snapshot index out of bound.");

    // Check MIMO coefficients
    bool add_coefficients = false;
    arma::uword n_path_add = 0ULL;
    if (coeff_re_add != nullptr)
    {
        if (this->coeff_re.size() == 0)
            throw std::invalid_argument("Channel object requires exisiting MIMO coefficients.");

        if (coeff_im_add == nullptr || delay_add == nullptr)
            throw std::invalid_argument("'coeff_im_add' or 'delay_add' is missing.");

        n_path_add = coeff_re_add->n_slices;
        if (coeff_im_add->n_slices != n_path_add)
            throw std::invalid_argument("Size of 'coeff_im_add' does not match size of 'coeff_re_add'");
        if (delay_add->n_slices != n_path_add)
            throw std::invalid_argument("Size of 'delay_add' does not match size of 'coeff_re_add'");

        arma::uword n_rx = this->n_rx(), n_tx = this->n_tx();
        if (coeff_re_add->n_rows != n_rx || coeff_re_add->n_cols != n_tx ||
            coeff_im_add->n_rows != n_rx || coeff_im_add->n_cols != n_tx)
            throw std::invalid_argument("MIMO matrix size mismatch.");

        n_rx = this->delay[i_snap].n_rows, n_tx = this->delay[i_snap].n_cols;
        if (delay_add->n_rows != n_rx || delay_add->n_cols != n_tx)
            throw std::invalid_argument("Delay matrix size mismatch.");

        add_coefficients = true;
    }
    if (!add_coefficients && (coeff_im_add != nullptr || delay_add != nullptr))
        throw std::invalid_argument("'coeff_re_add' is missing.");
    if (!add_coefficients && this->coeff_re.size() != 0)
        throw std::invalid_argument("The channel object has coefficients, but no coefficients are added!");

    // Check interaction coordinates
    bool add_interact_coord = false;
    if (no_interact_add != nullptr)
    {
        if (this->no_interact.size() == 0)
            throw std::invalid_argument("Channel object requires exisiting interaction coordinates.");

        if (interact_coord_add == nullptr)
            throw std::invalid_argument("'no_interact_add' is given but 'interact_coord_add' is missing.");

        n_path_add = (n_path_add == 0ULL) ? no_interact_add->n_elem : n_path_add;
        if (no_interact_add->n_elem != n_path_add)
            throw std::invalid_argument("Size of 'no_interact_add' does not match size of 'coeff_re_add'");

        auto sum_no_interact = (arma::uword)arma::sum(*no_interact_add);
        if (interact_coord_add->n_cols != sum_no_interact)
            throw std::invalid_argument("Number of columns in 'interact_coord_add' must match sum of 'no_interact_add'.");

        add_interact_coord = true;
    }
    if (!add_interact_coord && interact_coord_add != nullptr)
        throw std::invalid_argument("'interact_coord_add' is given but 'no_interact_add' is missing.");
    if (!add_interact_coord && this->no_interact.size() != 0)
        throw std::invalid_argument("The channel object has interaction coordinates, but no coordinates are added!");

    // Check path gain
    if (path_gain_add != nullptr)
    {
        if (this->path_gain.size() == 0)
            throw std::invalid_argument("Channel object requires exisiting 'path_gain'.");

        n_path_add = (n_path_add == 0ULL) ? path_gain_add->n_elem : n_path_add;
        if (path_gain_add->n_elem != n_path_add)
            throw std::invalid_argument("Size of 'path_gain_add' does not match 'n_path_add'");
    }
    else if (this->path_gain.size() != 0)
        throw std::invalid_argument("The channel object has 'path_gain', but no 'path_gain_add' is provided!");

    // Check path_length
    if (path_length_add != nullptr)
    {
        if (this->path_length.size() == 0)
            throw std::invalid_argument("Channel object requires exisiting 'path_length'.");

        n_path_add = (n_path_add == 0ULL) ? path_length_add->n_elem : n_path_add;
        if (path_length_add->n_elem != n_path_add)
            throw std::invalid_argument("Size of 'path_length_add' does not match 'n_path_add'");
    }
    else if (this->path_length.size() != 0)
        throw std::invalid_argument("The channel object has 'path_length', but no 'path_length_add' is provided!");

    // Check path_polarization
    if (path_polarization_add != nullptr)
    {
        if (this->path_polarization.size() == 0)
            throw std::invalid_argument("Channel object requires exisiting 'path_polarization'.");

        n_path_add = (n_path_add == 0ULL) ? path_polarization_add->n_cols : n_path_add;
        if (path_polarization_add->n_cols != n_path_add)
            throw std::invalid_argument("Number of columns in 'path_polarization_add' does not match 'n_path_add'");
        if (path_polarization_add->n_rows != 8ULL)
            throw std::invalid_argument("Number of rows in 'path_polarization_add' must be 8");
    }
    else if (this->path_polarization.size() != 0)
        throw std::invalid_argument("The channel object has 'path_polarization', but no 'path_polarization_add' is provided!");

    // Check path_angles
    if (path_angles_add != nullptr)
    {
        if (this->path_angles.size() == 0)
            throw std::invalid_argument("Channel object requires exisiting 'path_angles'.");

        n_path_add = (n_path_add == 0ULL) ? path_angles_add->n_rows : n_path_add;
        if (path_angles_add->n_rows != n_path_add)
            throw std::invalid_argument("Number of rows in 'path_angles_add' does not match 'n_path_add'");
        if (path_angles_add->n_cols != 4ULL)
            throw std::invalid_argument("Number of columns in 'path_angles_add' must be 4");
    }
    else if (this->path_angles.size() != 0)
        throw std::invalid_argument("The channel object has 'path_angles', but no 'path_angles_add' is provided!");

    // Check path_fbs_pos
    if (path_fbs_pos_add != nullptr)
    {
        if (this->path_fbs_pos.size() == 0)
            throw std::invalid_argument("Channel object requires exisiting 'path_fbs_pos'.");

        n_path_add = (n_path_add == 0ULL) ? path_fbs_pos_add->n_cols : n_path_add;
        if (path_fbs_pos_add->n_cols != n_path_add)
            throw std::invalid_argument("Number of columns in 'path_fbs_pos_add' does not match 'n_path_add'");
        if (path_fbs_pos_add->n_rows != 3ULL)
            throw std::invalid_argument("Number of rows in 'path_fbs_pos_add' must be 3");
    }
    else if (this->path_fbs_pos.size() != 0)
        throw std::invalid_argument("The channel object has 'path_fbs_pos', but no 'path_fbs_pos_add' is provided!");

    // Check path_lbs_pos
    if (path_lbs_pos_add != nullptr)
    {
        if (this->path_lbs_pos.size() == 0)
            throw std::invalid_argument("Channel object requires exisiting 'path_lbs_pos'.");

        n_path_add = (n_path_add == 0ULL) ? path_lbs_pos_add->n_cols : n_path_add;
        if (path_lbs_pos_add->n_cols != n_path_add)
            throw std::invalid_argument("Number of columns in 'path_lbs_pos_add' does not match 'n_path_add'");
        if (path_lbs_pos_add->n_rows != 3ULL)
            throw std::invalid_argument("Number of rows in 'path_lbs_pos_add' must be 3");
    }
    else if (this->path_lbs_pos.size() != 0)
        throw std::invalid_argument("The channel object has 'path_lbs_pos', but no 'path_lbs_pos_add' is provided!");

    // Add data to existing channel
    if (add_coefficients)
    {
        this->coeff_re[i_snap] = arma::join_slices(this->coeff_re[i_snap], *coeff_re_add);
        this->coeff_im[i_snap] = arma::join_slices(this->coeff_im[i_snap], *coeff_im_add);
        this->delay[i_snap] = arma::join_slices(this->delay[i_snap], *delay_add);
    }

    if (add_interact_coord)
    {
        this->no_interact[i_snap] = arma::join_cols(this->no_interact[i_snap], *no_interact_add);
        this->interact_coord[i_snap] = arma::join_rows(this->interact_coord[i_snap], *interact_coord_add);
    }

    if (path_gain_add != nullptr)
        this->path_gain[i_snap] = arma::join_cols(this->path_gain[i_snap], *path_gain_add);

    if (path_length_add != nullptr)
        this->path_length[i_snap] = arma::join_cols(this->path_length[i_snap], *path_length_add);

    if (path_polarization_add != nullptr)
        this->path_polarization[i_snap] = arma::join_rows(this->path_polarization[i_snap], *path_polarization_add);

    if (path_angles_add != nullptr)
        this->path_angles[i_snap] = arma::join_cols(this->path_angles[i_snap], *path_angles_add);

    if (path_fbs_pos_add != nullptr)
        this->path_fbs_pos[i_snap] = arma::join_rows(this->path_fbs_pos[i_snap], *path_fbs_pos_add);

    if (path_lbs_pos_add != nullptr)
        this->path_lbs_pos[i_snap] = arma::join_rows(this->path_lbs_pos[i_snap], *path_lbs_pos_add);
}

/*!MD
# .calc_effective_path_gain
Calculate the effective path gain per snapshot in linear scale

## Description:
- Sums power over all paths and TX/RX antenna pairs to produce one gain value per snapshot
- Uses `coeff_re`/`coeff_im` if available; falls back to `path_polarization` assuming ideal XPOL antennas
- Throws if neither coefficients nor polarization data are present
- Member function of [[channel]]
- Allowed datatypes: `float` or `double`

## Declaration:
```
arma::Col<dtype> quadriga_lib::channel<dtype>::calc_effective_path_gain(bool assume_valid = false) const;
```

## Input Arguments:
- **`assume_valid`** *(optional)* — Skip internal consistency checks for performance in trusted contexts

## Returns:
- Effective path gains in linear scale, one entry per snapshot, `[n_snap]`
MD!*/

template <typename dtype>
arma::Col<dtype> quadriga_lib::channel<dtype>::calc_effective_path_gain(bool assume_valid) const
{
    arma::uword n_snap_arma = this->n_snap();
    size_t n_snap_vector = (size_t)n_snap_arma;

    if (n_snap_arma == 0ULL)
        return arma::Col<dtype>();

    if (coeff_re.empty() && path_polarization.empty())
        throw std::invalid_argument("Neither coefficients nor polarization are provided.");

    // Check if data is valid
    if (!assume_valid)
    {
        std::string error_message = this->is_valid();
        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());
    }

    bool use_coeff = !coeff_re.empty();
    bool has_path_gain = !path_gain.empty();
    bool has_path_length = !path_length.empty();
    bool has_multi_freq = center_frequency.n_elem == n_snap_arma;

    dtype fGHz_PG = !center_frequency.is_empty() ? center_frequency.at(0) * (dtype)1.0e-9 : (dtype)1.0;
    fGHz_PG = -(dtype)32.45 - (dtype)20.0 * std::log10(fGHz_PG);

    auto n_paths = this->n_path();
    auto p_path = n_paths.memptr();
    arma::Col<dtype> PG(n_snap_arma);
    auto pPG = PG.memptr();

    for (size_t iS = 0; iS < n_snap_vector; ++iS)
        if (p_path[iS] != 0)
        {
            dtype p = (dtype)0.0;
            if (use_coeff)
            {
                for (auto &a : coeff_re[iS])
                    p += a * a;
                for (auto &a : coeff_im[iS])
                    p += a * a;
            }
            else // use polarization
            {
                const dtype *p_pol = path_polarization[iS].memptr();
                const dtype *p_gain = has_path_gain ? path_gain[iS].memptr() : nullptr;
                const dtype *p_length = has_path_length ? path_length[iS].memptr() : nullptr;

                if (has_multi_freq)
                    fGHz_PG = -(dtype)32.45 - (dtype)20.0 * std::log10(center_frequency.at(iS) * (dtype)1.0e-9);

                for (arma::uword iP = 0ULL; iP < p_path[iS]; ++iP)
                {
                    dtype gain = (dtype)1.0;
                    if (has_path_gain) // Use provided gain
                        gain = std::sqrt(p_gain[iP]);
                    else if (has_path_length) // Calculate FSPL
                    {
                        gain = fGHz_PG - (dtype)20.0 * std::log10(p_length[iP]);
                        gain = std::pow((dtype)10.0, (dtype)0.1 * gain);
                        gain = std::sqrt(gain);
                    }
                    for (unsigned j = 0; j < 8; ++j)
                    {
                        dtype v = gain * p_pol[8 * iP + j];
                        p += v * v;
                    }
                }
            }
            pPG[iS] = p;
        }

    return PG;
}

/*!MD
# .write_paths_to_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

## Description:
- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for Blender)
- Tubes are color-coded by path gain using a selected colormap; radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` limits total count
- Member function of [[channel]]
- Allowed datatypes: `float` or `double`

## Declaration:
```
void quadriga_lib::channel<dtype>::write_paths_to_obj_file(
    std::string fn,
    arma::uword max_no_paths = 0,
    dtype gain_max = -60.0,
    dtype gain_min = -140.0,
    std::string colormap = "jet",
    arma::uvec i_snap = {},
    dtype radius_max = 0.05,
    dtype radius_min = 0.01,
    arma::uword n_edges = 5) const;
```

## Input Arguments:
- **`fn`** — Output `.obj` file path
- **`max_no_paths`** *(optional)* — Max paths to export; `0` includes all paths above `gain_min`
- **`gain_max`** *(optional)* — Upper gain threshold in dB for color/radius mapping; higher values are clipped
- **`gain_min`** *(optional)* — Lower gain threshold in dB; paths below this are excluded
- **`colormap`** *(optional)* — Colormap name; see [[colormap]] for supported options
- **`i_snap`** *(optional)* — 0-based snapshot indices to include; empty exports all snapshots
- **`radius_max`** *(optional)* — Tube radius in meters at maximum gain
- **`radius_min`** *(optional)* — Tube radius in meters at minimum gain
- **`n_edges`** *(optional)* — Vertices per tube cross-section; must be ≥ 3

## See also:
- [[path_to_tube]] (generates tube geometry from path data)
- [[colormap]] (colormap lookup used for coloring)
MD!*/

template <typename dtype>
void quadriga_lib::channel<dtype>::write_paths_to_obj_file(std::string fn,
                                                           arma::uword max_no_paths,
                                                           dtype gain_max, dtype gain_min,
                                                           std::string colormap,
                                                           arma::uvec i_snap,
                                                           dtype radius_max, dtype radius_min,
                                                           arma::uword n_edges) const
{

    // Check validity
    std::string error_message = this->is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (this->empty())
        throw std::invalid_argument("Channel contains no data.");

    if (this->center_frequency.empty())
        throw std::invalid_argument("Center frequency is missing.");

    if (this->no_interact.empty() || this->interact_coord.empty())
        throw std::invalid_argument("Ray tracing data (no_interact, interact_coord) is missing.");

    if (this->path_polarization.empty() && this->coeff_re.empty())
        throw std::invalid_argument("MIMO coefficients or path-metadata is missing.");

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

    // Extract the file name from the path
    std::string fn_mtl_base;
    arma::uword pos = fn_mtl.find_last_of("/");
    if (pos != std::string::npos)
        fn_mtl_base = fn_mtl.substr(pos + 1ULL);
    else
        fn_mtl_base = fn_mtl;

    arma::uword n_snap_in = this->n_snap();
    arma::uword n_rx = this->n_rx();
    arma::uword n_tx = this->n_tx();

    if (i_snap.n_elem == 0ULL)
        i_snap = arma::regspace<arma::uvec>(0ULL, n_snap_in - 1ULL);

    if (arma::any(i_snap >= n_snap_in))
        throw std::invalid_argument("Snapshot indices 'i_snap' cannot exceed the number of snapshots in the channel.");

    arma::uword n_snap_out = i_snap.n_elem;

    if (radius_min < (dtype)0.0 || radius_max < (dtype)0.0)
        throw std::invalid_argument("Radius cannot be negative.");

    // Make sure that the minimum radius is smaller than the maximum
    radius_min = (radius_min < radius_max) ? radius_min : radius_max;

    // Colormap
    arma::uchar_mat cmap = quadriga_lib::colormap(colormap);
    arma::uword n_cmap = (arma::uword)cmap.n_rows;

    // Export colormap to material file
    std::ofstream outFile(fn_mtl);
    if (outFile.is_open())
    {
        // Write some text to the file
        outFile << "# QuaDRiGa " << "path data colormap\n\n";
        for (arma::uword i = 0ULL; i < n_cmap; ++i)
        {
            double R = (double)cmap(i, 0ULL) / 255.0;
            double G = (double)cmap(i, 1ULL) / 255.0;
            double B = (double)cmap(i, 2ULL) / 255.0;
            outFile << "newmtl QuaDRiGa_PATH_" << colormap << "_" << std::setfill('0') << std::setw(2) << i << "\n";
            outFile << std::fixed << std::setprecision(6) << "Kd " << R << " " << G << " " << B << "\n\n";
        }
        outFile.close();
    }
    else
        throw std::invalid_argument("Could not write material file.");

    // Export paths to OBJ file
    outFile = std::ofstream(fn);
    if (outFile.is_open())
    {
        // Write some text to the file
        outFile << "# QuaDRiGa Path OBJ File\n";
        outFile << "mtllib " << fn_mtl_base << "\n";
    }
    else
        throw std::invalid_argument("Could not write OBJ file.");

    bool moving_tx = this->tx_pos.n_cols != 1ULL;
    bool moving_rx = this->rx_pos.n_cols != 1ULL;

    // Export each snapshot
    arma::uword vert_counter = 0ULL;
    for (arma::uword i_snap_out = 0ULL; i_snap_out < n_snap_out; ++i_snap_out)
    {
        arma::uword i_snap_in = i_snap(i_snap_out);

        // Extract path coordinates
        dtype tx = moving_tx ? this->tx_pos(0ULL, i_snap_in) : this->tx_pos(0ULL, 0ULL);
        dtype ty = moving_tx ? this->tx_pos(1ULL, i_snap_in) : this->tx_pos(1ULL, 0ULL);
        dtype tz = moving_tx ? this->tx_pos(2ULL, i_snap_in) : this->tx_pos(2ULL, 0ULL);

        dtype gainF = (this->center_frequency.n_elem > 1ULL) ? this->center_frequency(i_snap_in) : this->center_frequency(0ULL);
        gainF = (dtype)-32.45 - (dtype)20.0 * std::log10(gainF * (dtype)1.0e-9);

        dtype rx = moving_rx ? this->rx_pos(0ULL, i_snap_in) : this->rx_pos(0ULL, 0ULL);
        dtype ry = moving_rx ? this->rx_pos(1ULL, i_snap_in) : this->rx_pos(1ULL, 0ULL);
        dtype rz = moving_rx ? this->rx_pos(2ULL, i_snap_in) : this->rx_pos(2ULL, 0ULL);

        // Calculate path coordinates
        arma::Col<dtype> path_length;
        std::vector<arma::Mat<dtype>> path_coord;
        quadriga_lib::coord2path<dtype>(tx, ty, tz, rx, ry, rz, &this->no_interact[i_snap_in], &this->interact_coord[i_snap_in],
                                        &path_length, nullptr, nullptr, nullptr, &path_coord);

        // Calculate path power
        arma::uword n_path = path_coord.size();
        arma::Col<dtype> path_power_dB(n_path);
        arma::s32_vec color_index(n_path);

        dtype scl = (dtype)63.0 / (gain_max - gain_min);
        for (arma::uword i_path = 0ULL; i_path < n_path; ++i_path)
        {
            // Lambda to calculate the power and check if the pattern has any entry > -200 dB
            auto calc_power = [](const dtype *Re, const dtype *Im, arma::uword n_elem) -> dtype
            {
                dtype p = dtype(0.0);
                for (arma::uword i = 0ULL; i < n_elem; ++i)
                    p += Re[i] * Re[i] + Im[i] * Im[i];
                return (dtype)10.0 * std::log10(p);
                ;
            };

            dtype p = (dtype)0.0;
            if (!this->coeff_re.empty()) // Use MIMO coefficients
                p = calc_power(this->coeff_re[i_snap_in].slice_memptr(i_path), this->coeff_im[i_snap_in].slice_memptr(i_path), n_rx * n_tx);
            else if (!this->path_polarization.empty()) // Use XPR
            {
                const dtype *p_xpr = this->path_polarization[i_snap_in].colptr(i_path);
                for (arma::uword m = 0ULL; m < 8ULL; ++m) // sum( xprmat(:,n).^2 )
                    p += *p_xpr * *p_xpr, ++p_xpr;
                p = (dtype)10.0 * std::log10(p);
                p += gainF - (dtype)20.0 * std::log10(path_length(i_path));
            }
            path_power_dB(i_path) = p;

            dtype x = p;
            x = (x < gain_min) ? gain_min : x;
            x = (x > gain_max) ? gain_max : x;
            x = x - gain_min;
            x = x * scl;
            x = (x > (dtype)63.0) ? (dtype)63.0 : x;
            x = std::round(x);

            color_index(i_path) = (p < gain_min) ? -1 : (int)x;
        }

        // Sort the path power in decreasing order and return the indices
        arma::uvec pow_indices = arma::sort_index(path_power_dB, "descend");
        arma::uword *i_pow_sorted = pow_indices.memptr();

        // Limit the number of paths that should be shown
        if (max_no_paths != 0ULL)
            for (arma::uword i = max_no_paths; i < n_path; ++i)
                color_index(i_pow_sorted[i]) = -1;

        // Write some descriptive information
        outFile << "\n# Snapshot " << i_snap_in << "\n";
        outFile << "#  No.   Length[m]     Gain[dB]   ID" << "\n";

        for (arma::uword i_path = 0ULL; i_path < n_path; ++i_path)
        {
            arma::uword i_path_sorted = i_pow_sorted[i_path];
            dtype l = path_length(i_path_sorted);
            dtype p = path_power_dB(i_path_sorted);

            if (p > (dtype)-200.0)
            {
                outFile << "# " << std::setfill('0') << std::setw(4) << i_path << " ";
                if (l < 1000.0f)
                    outFile << " ";
                if (l < 100.0f)
                    outFile << " ";
                if (l < 10.0f)
                    outFile << " ";
                outFile << std::fixed << std::setprecision(6) << l << "  ";
                if (p > -100.0f)
                    outFile << " ";
                if (p > -10.0f)
                    outFile << " ";
                if (p > 10.0f)
                    outFile << " ";
                else if (p > 0.0f)
                    outFile << "  ";

                outFile << std::fixed << std::setprecision(6) << p;

                if (color_index(i_path_sorted) >= 10)
                    outFile << "   " << color_index(i_path_sorted);
                else if (color_index(i_path_sorted) >= 0)
                    outFile << "    " << color_index(i_path_sorted);

                outFile << "\n";
            }
        }

        // Write OBJ elements
        scl = radius_max / (gain_max - gain_min);
        for (arma::uword i_path = 0ULL; i_path < n_path; ++i_path)
        {
            arma::uword i_path_sorted = i_pow_sorted[i_path];
            if (color_index(i_path_sorted) >= 0)
            {
                // Calculate radius
                dtype p = path_power_dB(i_path_sorted);
                dtype radius = p - gain_min;
                radius = radius * scl;
                radius = (radius > radius_max) ? radius_max : radius;
                radius = (radius < radius_min) ? radius_min : radius;

                // Write object name to OBJ file
                outFile << "\no QuaDRiGa_path_s" << std::setfill('0') << std::setw(4) << i_snap_in << "_p" << std::setfill('0') << std::setw(4) << i_path << "\n";

                ++vert_counter;
                if (std::abs(radius) > (dtype)1.0e-4)
                {
                    // Calculate vertices and faces
                    arma::Mat<dtype> vert;
                    arma::umat faces;
                    quadriga_lib::path_to_tube(&path_coord[i_path_sorted], &vert, &faces, radius, (arma::uword)n_edges);

                    // Write vertices to file
                    arma::uword n_vert = vert.n_cols;
                    for (arma::uword iV = 0ULL; iV < n_vert; ++iV)
                        outFile << std::defaultfloat << "v " << vert(0ULL, iV) << " " << vert(1ULL, iV) << " " << vert(2ULL, iV) << "\n";

                    outFile << "usemtl QuaDRiGa_PATH_" << colormap << "_" << std::setfill('0') << std::setw(2) << color_index(i_path_sorted) << "\n";

                    // Write faces to file
                    arma::uword n_faces = faces.n_cols;
                    for (arma::uword iF = 0ULL; iF < n_faces; ++iF)
                        outFile << "f " << faces(0ULL, iF) + vert_counter << " " << faces(1ULL, iF) + vert_counter << " " << faces(2ULL, iF) + vert_counter << " " << faces(3ULL, iF) + vert_counter << "\n";

                    vert_counter += vert.n_cols - 1ULL;
                }
                else
                {
                    // Write vertices to file
                    arma::uword n_vert = path_coord[i_path_sorted].n_cols;
                    for (arma::uword iV = 0ULL; iV < n_vert; ++iV)
                        outFile << std::defaultfloat << "v " << path_coord[i_path_sorted](0ULL, iV) << " " << path_coord[i_path_sorted](1ULL, iV) << " " << path_coord[i_path_sorted](2ULL, iV) << "\n";

                    outFile << "usemtl QuaDRiGa_PATH_" << colormap << "_" << std::setfill('0') << std::setw(2) << color_index(i_path_sorted) << "\n";

                    for (arma::uword iV = 0ULL; iV < n_vert - 1ULL; ++iV)
                        outFile << "l " << vert_counter << " " << vert_counter + 1ULL << "\n", ++vert_counter;
                }
            }
        }
    }
    outFile.close();
}

// Instantiate templates
template class quadriga_lib::channel<float>;
template class quadriga_lib::channel<double>;
