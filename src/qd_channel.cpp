// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "qd_helpers_shared.hpp"

/*!SECTION
Channel class
SECTION!*/

/*!MD
# channel<++>
Class for storing and managing MIMO channel data and metadata across multiple snapshots

- Represents path-level MIMO channel data between antenna arrays over multiple time snapshots
- Each snapshot may have a different number of propagation paths `n_path`
- Unstructured metadata supported via `par_names` / `par_data`

## Attributes:<br>
| Attribute                                         | Size                                                            | Description                                                 |
| ------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| `std::string name`                                | —                                                               | Name of the channel object                                  |
| `arma::Col<dtype> center_freq`               | `[1]`; `[n_snap]`, or `[]`                                      | Center frequency                                            |
| `arma::Mat<dtype> tx_pos`                         | `[3, n_snap]` or `[3, 1]` = static                              | Transmitter positions                                       |
| `arma::Mat<dtype> rx_pos`                         | `[3, n_snap]` or `[3, 1]` = static                              | Receiver positions                                          |
| `arma::Mat<dtype> tx_orientation`                 | `[3, n_snap]`; `[3, 1]` = static, or `[]` = no rotation         | Transmitter orientation (Euler angles)                      |
| `arma::Mat<dtype> rx_orientation`                 | `[3, n_snap]`; `[3, 1]` = static, or `[]` = no rotation         | Receiver orientation (Euler angles)                         |
| `std::vector<arma::Cube<dtype>> coeff_re`         | per snap `[n_rx, n_tx, n_path]`                                 | Channel coefficients, real part                             |
| `std::vector<arma::Cube<dtype>> coeff_im`         | per snap `[n_rx, n_tx, n_path]`                                 | Channel coefficients, imaginary part                        |
| `std::vector<arma::Cube<dtype>> delay`            | per snap `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` = broadcast | Path delays in seconds                                      |
| `std::vector<arma::Col<dtype>> path_gain`         | per snap `[n_path]`                                             | Path gains before antenna pattern                           |
| `std::vector<arma::Col<dtype>> path_length`       | per snap `[n_path]`                                             | Path lengths TX to RX                                       |
| `std::vector<arma::Mat<dtype>> polmat`            | per snap `[8, n_path]`                                          | Interleaved polarization transfer matrices                  |
| `std::vector<arma::Mat<dtype>> path_angles`       | per snap `[n_path, 4]`                                          | Angles {AOD, EOD, AOA, EOA} in rad                          |
| `std::vector<arma::Mat<dtype>> fbs_pos`           | per snap `[3, n_path]`                                          | First-bounce scatterer positions                            |
| `std::vector<arma::Mat<dtype>> lbs_pos`           | per snap `[3, n_path]`                                          | Last-bounce scatterer positions                             |
| `std::vector<arma::Col<unsigned>> n_interact`     | per snap `[n_path]`                                             | Number of interactions per path                             |
| `std::vector<arma::Mat<dtype>> interact_coord`    | per snap `[3, sum(n_interact)]`                                 | Interaction point coordinates                               |
| `std::vector<std::string> par_names`              | —                                                               | Names of unstructured metadata fields                       |
| `std::vector<std::any> par_data`                  | —                                                               | Unstructured metadata values (string, scalar, matrix, etc.) |
| `int initial_position`                            | scalar                                                          | 0-based index of the reference snapshot                     |

## Simple member functions:<br>
| Method        | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| `.n_snap()`   | Returns the number of snapshots                               |
| `.n_rx()`     | Returns number of receive antennas; 0 if coefficients absent  |
| `.n_tx()`     | Returns number of transmit antennas; 0 if coefficients absent |
| `.n_path()`   | Returns number of paths per snapshot as a vector              |
| `.empty()`    | Returns true if the object contains no channel data           |
| `.is_valid()` | Returns empty string if valid, otherwise an error message     |

## Complex member functions:
- .[[add_paths]]

Computations and file I/O on a channel are free functions, see section "Channel functions"
(e.g. [[channel_calc_effective_path_gain]], [[channel_export_obj_file]], [[hdf5_write_channel]], [[quantize_delays]]).
MD!*/

// Object dimensions
template <typename dtype>
arma::uword quadriga_lib::channel<dtype>::n_snap() const
{
    arma::uword s_arma = 0ULL;

    if (center_freq.n_elem > s_arma) // 1 or s
        s_arma = center_freq.n_elem;

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

    if (polmat.size() > s_vector)
        s_vector = polmat.size();

    if (path_angles.size() > s_vector)
        s_vector = path_angles.size();

    if (fbs_pos.size() > s_vector)
        s_vector = fbs_pos.size();

    if (lbs_pos.size() > s_vector)
        s_vector = lbs_pos.size();

    if (n_interact.size() > s_vector)
        s_vector = n_interact.size();

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
    else if (n_interact.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = n_interact[i].n_elem;
    else if (path_gain.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_gain[i].n_elem;
    else if (path_length.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_length[i].n_elem;
    else if (polmat.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = polmat[i].n_cols;
    else if (path_angles.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_angles[i].n_rows;
    else if (fbs_pos.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = fbs_pos[i].n_cols;
    else if (lbs_pos.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = lbs_pos[i].n_cols;
    else if (delay.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = delay[i].n_slices;

    return n_path;
}

// Returns true if the channel object contains no data
template <typename dtype>
bool quadriga_lib::channel<dtype>::empty() const
{
    if (center_freq.n_elem != 0ULL)
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

    if (polmat.size() != 0)
        return false;

    if (path_angles.size() != 0)
        return false;

    if (fbs_pos.size() != 0)
        return false;

    if (lbs_pos.size() != 0)
        return false;

    if (n_interact.size() != 0)
        return false;

    if (interact_coord.size() != 0)
        return false;

    if (initial_position != 0)
        return false;

    return true;
}

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

    if (center_freq.n_elem != 0ULL && center_freq.n_elem != 1ULL && center_freq.n_elem != n_snap_arma)
        return std::string("Number of entries in 'center_freq' must be 0, 1 or match the number of snapshots.");

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
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'coeff_re'.";

            if (coeff_im[i].n_rows != n_rx || coeff_im[i].n_cols != n_tx || coeff_im[i].n_slices != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'coeff_im'.";

            if (!(delay[i].n_rows == n_rx && delay[i].n_cols == n_tx && delay[i].n_slices == n_pth[i]) &&
                !(delay[i].n_rows == 1 && delay[i].n_cols == 1 && delay[i].n_slices == n_pth[i]))
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'delay'.";
        }
    }
    else if (!coeff_im.empty())
        return std::string("Real part of channel coefficients 'coeff_re' is missing or incomplete.");
    else if (!delay.empty())
    {
        if (delay.size() != n_snap_vector)
            return std::string("'delay' must be empty or match the number of snapshots.");

        for (size_t i = 0; i < n_snap_vector; ++i)
            if (delay[i].n_rows != 1 || delay[i].n_cols != 1 || delay[i].n_slices != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'delay'.";
    }

    if (path_gain.size() != 0 && path_gain.size() != n_snap_vector)
        return std::string("'path_gain' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_gain.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_gain[i].n_elem != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'path_gain'.";

    if (path_length.size() != 0 && path_length.size() != n_snap_vector)
        return std::string("'path_length' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_length.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_length[i].n_elem != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'path_length'.";

    if (polmat.size() != 0 && polmat.size() != n_snap_vector)
        return std::string("'polmat' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && polmat.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (polmat[i].n_rows != 8ULL || polmat[i].n_cols != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'polmat'.";

    if (path_angles.size() != 0 && path_angles.size() != n_snap_vector)
        return std::string("'path_angles' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_angles.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_angles[i].n_rows != n_pth[i] || path_angles[i].n_cols != 4ULL)
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'path_angles'.";

    if (fbs_pos.size() != 0 && fbs_pos.size() != n_snap_vector)
        return std::string("'fbs_pos' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && fbs_pos.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (fbs_pos[i].n_rows != 3ULL || fbs_pos[i].n_cols != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'fbs_pos'.";

    if (lbs_pos.size() != 0 && lbs_pos.size() != n_snap_vector)
        return std::string("'lbs_pos' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && lbs_pos.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (lbs_pos[i].n_rows != 3ULL || lbs_pos[i].n_cols != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'lbs_pos'.";

    if (n_interact.size() != 0 && n_interact.size() != n_snap_vector)
        return std::string("'n_interact' must be empty or match the number of snapshots.");
    else if (n_interact.size() == 0 && interact_coord.size() != 0)
        return std::string("'interact_coord' is provided but 'n_interact' is missing.");
    else if (n_interact.size() == n_snap_vector && interact_coord.size() != n_snap_vector)
        return std::string("'n_interact' is provided but 'interact_coord' is missing or has wrong number of snapshots.");
    else if (n_interact.size() == n_snap_vector && interact_coord.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
        {
            if (n_interact[i].n_elem != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'n_interact'.";

            if (interact_coord[i].n_rows != 3ULL)
                return "Snapshot " + std::to_string(i) + ": 'interact_coord' must have 3 rows.";

            arma::uword cnt = 0;
            for (const unsigned *p = n_interact[i].begin(); p < n_interact[i].end(); ++p)
                cnt += (arma::uword)*p;

            if (interact_coord[i].n_cols != cnt)
                return "Snapshot " + std::to_string(i) + ": Number of columns in 'interact_coord' must match the sum of 'n_interact'.";
        }

    if (par_names.size() != par_data.size())
        return std::string("Number of elements in 'par_data' must match number of elements in 'par_names'.");

    for (size_t i = 0; i < par_names.size(); ++i)
        if (any_type_id(&par_data[i]) < 0)
            return std::string("Unsupported datatype in unstructured data.");

    return std::string("");
}

/*!MD
# .add_paths
Append new propagation paths to an existing channel snapshot

- Adds path-level data to snapshot `i_snap` in a `channel` object; does not modify `tx_pos`, `rx_pos`, or orientation fields
- All provided fields must have consistent length `n_path_add` and match existing snapshot structure
- The first provided field sets `n_path_add`, also when it holds 0 paths; a field with a different path count throws

## Declaration:
```
void quadriga_lib::channel<dtype>::add_paths(
    arma::uword i_snap,
    const arma::Cube<dtype> *coeff_re_add = nullptr,
    const arma::Cube<dtype> *coeff_im_add = nullptr,
    const arma::Cube<dtype> *delay_add = nullptr,
    const arma::u32_vec *n_interact_add = nullptr,
    const arma::Mat<dtype> *interact_coord_add = nullptr,
    const arma::Col<dtype> *path_gain_add = nullptr,
    const arma::Col<dtype> *path_length_add = nullptr,
    const arma::Mat<dtype> *polmat_add = nullptr,
    const arma::Mat<dtype> *path_angles_add = nullptr,
    const arma::Mat<dtype> *fbs_pos_add = nullptr,
    const arma::Mat<dtype> *lbs_pos_add = nullptr);
```

## Inputs:
- **`i_snap`** — 0-based snapshot index to append paths to
- **`coeff_re_add`** *(optional)* — Real part of channel coefficients; `[n_rx, n_tx, n_path_add]`
- **`coeff_im_add`** *(optional)* — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_add]`
- **`delay_add`** *(optional)* — Propagation delays in seconds; `[n_rx, n_tx, n_path_add]` or `[1, 1, n_path_add]`
- **`n_interact_add`** *(optional)* — Number of interaction points per path; `[n_path_add]`
- **`interact_coord_add`** *(optional)* — Interaction point coordinates; `[3, sum(n_interact)]`
- **`path_gain_add`** *(optional)* — Path gains before antenna effects; `[n_path_add]`
- **`path_length_add`** *(optional)* — Path lengths from TX to RX phase center; `[n_path_add]`
- **`polmat_add`** *(optional)* — Interleaved polarization transfer matrices; `[8, n_path_add]`
- **`path_angles_add`** *(optional)* — Departure/arrival angles {AOD, EOD, AOA, EOA} in rad; `[n_path_add, 4]`
- **`fbs_pos_add`** *(optional)* — First-bounce scatterer positions; `[3, n_path_add]`
- **`lbs_pos_add`** *(optional)* — Last-bounce scatterer positions; `[3, n_path_add]`
MD!*/

template <typename dtype>
void quadriga_lib::channel<dtype>::add_paths(arma::uword i_snap,
                                             const arma::Cube<dtype> *coeff_re_add, const arma::Cube<dtype> *coeff_im_add, const arma::Cube<dtype> *delay_add,
                                             const arma::u32_vec *n_interact_add, const arma::Mat<dtype> *interact_coord_add,
                                             const arma::Col<dtype> *path_gain_add, const arma::Col<dtype> *path_length_add,
                                             const arma::Mat<dtype> *polmat_add, const arma::Mat<dtype> *path_angles_add,
                                             const arma::Mat<dtype> *fbs_pos_add, const arma::Mat<dtype> *lbs_pos_add)
{
    std::string error_message = this->is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    arma::uword n_snap_chan = this->n_snap();
    if (i_snap >= n_snap_chan)
        throw std::invalid_argument("Input 'i_snap' is out of range.");

    // Check MIMO coefficients
    // The first given field sets the number of added paths (also when it is 0), every other field must match
    bool add_coefficients = false;
    bool n_path_set = false;
    arma::uword n_path_add = 0ULL;
    if (coeff_re_add != nullptr)
    {
        if (this->coeff_re.size() == 0)
            throw std::invalid_argument("Channel object requires existing MIMO coefficients.");

        if (coeff_im_add == nullptr || delay_add == nullptr)
            throw std::invalid_argument("'coeff_im_add' or 'delay_add' is missing.");

        n_path_add = coeff_re_add->n_slices;
        n_path_set = true;
        if (coeff_im_add->n_slices != n_path_add)
            throw std::invalid_argument("Size of 'coeff_im_add' does not match size of 'coeff_re_add'.");
        if (delay_add->n_slices != n_path_add)
            throw std::invalid_argument("Size of 'delay_add' does not match size of 'coeff_re_add'.");

        arma::uword n_rx = this->n_rx(), n_tx = this->n_tx();
        if (coeff_re_add->n_rows != n_rx || coeff_re_add->n_cols != n_tx ||
            coeff_im_add->n_rows != n_rx || coeff_im_add->n_cols != n_tx)
            throw std::invalid_argument("Inputs 'coeff_re_add' and 'coeff_im_add' must have n_rx rows and n_tx columns.");

        n_rx = this->delay[i_snap].n_rows, n_tx = this->delay[i_snap].n_cols;
        if (delay_add->n_rows != n_rx || delay_add->n_cols != n_tx)
            throw std::invalid_argument("Input 'delay_add' must have the rows and columns of the snapshot's 'delay'.");

        add_coefficients = true;
    }
    if (!add_coefficients && (coeff_im_add != nullptr || delay_add != nullptr))
        throw std::invalid_argument("'coeff_re_add' is missing.");
    if (!add_coefficients && this->coeff_re.size() != 0)
        throw std::invalid_argument("The channel object has coefficients, but no coefficients are added.");

    // Check interaction coordinates
    bool add_interact_coord = false;
    if (n_interact_add != nullptr)
    {
        if (this->n_interact.size() == 0)
            throw std::invalid_argument("Channel object requires existing interaction coordinates.");

        if (interact_coord_add == nullptr)
            throw std::invalid_argument("'n_interact_add' is given but 'interact_coord_add' is missing.");

        if (!n_path_set)
            n_path_add = n_interact_add->n_elem, n_path_set = true;
        if (n_interact_add->n_elem != n_path_add)
            throw std::invalid_argument("Size of 'n_interact_add' does not match size of 'coeff_re_add'.");

        // 64-bit sum: arma::sum of a u32_vec is a 32-bit sum and can wrap
        arma::uword sum_no_interact = 0;
        for (const unsigned *p = n_interact_add->begin(); p < n_interact_add->end(); ++p)
            sum_no_interact += (arma::uword)*p;
        if (interact_coord_add->n_cols != sum_no_interact)
            throw std::invalid_argument("Number of columns in 'interact_coord_add' must match sum of 'n_interact_add'.");

        add_interact_coord = true;
    }
    if (!add_interact_coord && interact_coord_add != nullptr)
        throw std::invalid_argument("'interact_coord_add' is given but 'n_interact_add' is missing.");
    if (!add_interact_coord && this->n_interact.size() != 0)
        throw std::invalid_argument("The channel object has interaction coordinates, but no coordinates are added.");

    // Check path gain
    if (path_gain_add != nullptr)
    {
        if (this->path_gain.size() == 0)
            throw std::invalid_argument("Channel object requires existing 'path_gain'.");

        if (!n_path_set)
            n_path_add = path_gain_add->n_elem, n_path_set = true;
        if (path_gain_add->n_elem != n_path_add)
            throw std::invalid_argument("Size of 'path_gain_add' does not match 'n_path_add'.");
    }
    else if (this->path_gain.size() != 0)
        throw std::invalid_argument("The channel object has 'path_gain', but no 'path_gain_add' is provided.");

    // Check path_length
    if (path_length_add != nullptr)
    {
        if (this->path_length.size() == 0)
            throw std::invalid_argument("Channel object requires existing 'path_length'.");

        if (!n_path_set)
            n_path_add = path_length_add->n_elem, n_path_set = true;
        if (path_length_add->n_elem != n_path_add)
            throw std::invalid_argument("Size of 'path_length_add' does not match 'n_path_add'.");
    }
    else if (this->path_length.size() != 0)
        throw std::invalid_argument("The channel object has 'path_length', but no 'path_length_add' is provided.");

    // Check polmat
    if (polmat_add != nullptr)
    {
        if (this->polmat.size() == 0)
            throw std::invalid_argument("Channel object requires existing 'polmat'.");

        if (!n_path_set)
            n_path_add = polmat_add->n_cols, n_path_set = true;
        if (polmat_add->n_cols != n_path_add)
            throw std::invalid_argument("Number of columns in 'polmat_add' does not match 'n_path_add'.");
        if (polmat_add->n_rows != 8ULL)
            throw std::invalid_argument("Number of rows in 'polmat_add' must be 8.");
    }
    else if (this->polmat.size() != 0)
        throw std::invalid_argument("The channel object has 'polmat', but no 'polmat_add' is provided.");

    // Check path_angles
    if (path_angles_add != nullptr)
    {
        if (this->path_angles.size() == 0)
            throw std::invalid_argument("Channel object requires existing 'path_angles'.");

        if (!n_path_set)
            n_path_add = path_angles_add->n_rows, n_path_set = true;
        if (path_angles_add->n_rows != n_path_add)
            throw std::invalid_argument("Number of rows in 'path_angles_add' does not match 'n_path_add'.");
        if (path_angles_add->n_cols != 4ULL)
            throw std::invalid_argument("Number of columns in 'path_angles_add' must be 4.");
    }
    else if (this->path_angles.size() != 0)
        throw std::invalid_argument("The channel object has 'path_angles', but no 'path_angles_add' is provided.");

    // Check fbs_pos
    if (fbs_pos_add != nullptr)
    {
        if (this->fbs_pos.size() == 0)
            throw std::invalid_argument("Channel object requires existing 'fbs_pos'.");

        if (!n_path_set)
            n_path_add = fbs_pos_add->n_cols, n_path_set = true;
        if (fbs_pos_add->n_cols != n_path_add)
            throw std::invalid_argument("Number of columns in 'fbs_pos_add' does not match 'n_path_add'.");
        if (fbs_pos_add->n_rows != 3ULL)
            throw std::invalid_argument("Number of rows in 'fbs_pos_add' must be 3.");
    }
    else if (this->fbs_pos.size() != 0)
        throw std::invalid_argument("The channel object has 'fbs_pos', but no 'fbs_pos_add' is provided.");

    // Check lbs_pos
    if (lbs_pos_add != nullptr)
    {
        if (this->lbs_pos.size() == 0)
            throw std::invalid_argument("Channel object requires existing 'lbs_pos'.");

        if (!n_path_set)
            n_path_add = lbs_pos_add->n_cols, n_path_set = true;
        if (lbs_pos_add->n_cols != n_path_add)
            throw std::invalid_argument("Number of columns in 'lbs_pos_add' does not match 'n_path_add'.");
        if (lbs_pos_add->n_rows != 3ULL)
            throw std::invalid_argument("Number of rows in 'lbs_pos_add' must be 3.");
    }
    else if (this->lbs_pos.size() != 0)
        throw std::invalid_argument("The channel object has 'lbs_pos', but no 'lbs_pos_add' is provided.");

    // The fields of snapshot i_snap are written: an input must not be one of them or share memory with one
    auto check_mat = [&](const arma::Mat<dtype> *out, const char *name)
    {
        qd_check_alias(out, name, interact_coord_add, path_gain_add, path_length_add, polmat_add,
                       path_angles_add, fbs_pos_add, lbs_pos_add);
    };
    if (add_coefficients)
    {
        qd_check_alias(&this->coeff_re[i_snap], "coeff_re[i_snap]", coeff_re_add, coeff_im_add, delay_add);
        qd_check_alias(&this->coeff_im[i_snap], "coeff_im[i_snap]", coeff_re_add, coeff_im_add, delay_add);
        qd_check_alias(&this->delay[i_snap], "delay[i_snap]", coeff_re_add, coeff_im_add, delay_add);
    }
    if (add_interact_coord)
    {
        qd_check_alias(&this->n_interact[i_snap], "n_interact[i_snap]", n_interact_add);
        check_mat(&this->interact_coord[i_snap], "interact_coord[i_snap]");
    }
    if (path_gain_add != nullptr)
        check_mat(&this->path_gain[i_snap], "path_gain[i_snap]");
    if (path_length_add != nullptr)
        check_mat(&this->path_length[i_snap], "path_length[i_snap]");
    if (polmat_add != nullptr)
        check_mat(&this->polmat[i_snap], "polmat[i_snap]");
    if (path_angles_add != nullptr)
        check_mat(&this->path_angles[i_snap], "path_angles[i_snap]");
    if (fbs_pos_add != nullptr)
        check_mat(&this->fbs_pos[i_snap], "fbs_pos[i_snap]");
    if (lbs_pos_add != nullptr)
        check_mat(&this->lbs_pos[i_snap], "lbs_pos[i_snap]");

    // Add data to existing channel
    if (add_coefficients)
    {
        this->coeff_re[i_snap] = arma::join_slices(this->coeff_re[i_snap], *coeff_re_add);
        this->coeff_im[i_snap] = arma::join_slices(this->coeff_im[i_snap], *coeff_im_add);
        this->delay[i_snap] = arma::join_slices(this->delay[i_snap], *delay_add);
    }

    if (add_interact_coord)
    {
        this->n_interact[i_snap] = arma::join_cols(this->n_interact[i_snap], *n_interact_add);
        this->interact_coord[i_snap] = arma::join_rows(this->interact_coord[i_snap], *interact_coord_add);
    }

    if (path_gain_add != nullptr)
        this->path_gain[i_snap] = arma::join_cols(this->path_gain[i_snap], *path_gain_add);

    if (path_length_add != nullptr)
        this->path_length[i_snap] = arma::join_cols(this->path_length[i_snap], *path_length_add);

    if (polmat_add != nullptr)
        this->polmat[i_snap] = arma::join_rows(this->polmat[i_snap], *polmat_add);

    if (path_angles_add != nullptr)
        this->path_angles[i_snap] = arma::join_cols(this->path_angles[i_snap], *path_angles_add);

    if (fbs_pos_add != nullptr)
        this->fbs_pos[i_snap] = arma::join_rows(this->fbs_pos[i_snap], *fbs_pos_add);

    if (lbs_pos_add != nullptr)
        this->lbs_pos[i_snap] = arma::join_rows(this->lbs_pos[i_snap], *lbs_pos_add);
}

template class quadriga_lib::channel<float>;
template class quadriga_lib::channel<double>;
