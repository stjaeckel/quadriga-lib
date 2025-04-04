// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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
    if (name != "empty")
        return false;

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

// Add paths to an exisiting channel
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
            throw std::invalid_argument("Size of 'coeff_im_add' does not match size of 'coeff_re_add'");

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
            throw std::invalid_argument("Number of columns in 'no_interact_add' must match sum of 'no_interact_add'.");

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
            throw std::invalid_argument("Number of columns in 'path_angles_add' does not match 'n_path_add'");
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

// Calculate the the effective path gain (linear scale)
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

// Instantiate templates
template class quadriga_lib::channel<float>;
template class quadriga_lib::channel<double>;
