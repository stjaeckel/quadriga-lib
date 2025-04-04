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

#include "quadriga_tools.hpp"

// Convert path interaction coordinates into FBS/LBS positions, path length and angles
template <typename dtype>
void quadriga_lib::coord2path(dtype Tx, dtype Ty, dtype Tz, dtype Rx, dtype Ry, dtype Rz,
                              const arma::Col<unsigned> *no_interact, const arma::Mat<dtype> *interact_coord,
                              arma::Col<dtype> *path_length, arma::Mat<dtype> *fbs_pos, arma::Mat<dtype> *lbs_pos,
                              arma::Mat<dtype> *path_angles, std::vector<arma::Mat<dtype>> *path_coord, bool reverse_path)
{
    if (no_interact == nullptr)
        throw std::invalid_argument("Input 'no_interact' cannot be NULL.");

    size_t n_path = (size_t)no_interact->n_elem;

    // Calculate the total number of interactions
    unsigned interact_cnt = 0;
    const unsigned *p_interact = no_interact->memptr();
    for (size_t i = 0; i < n_path; ++i)
        interact_cnt += p_interact[i];

    size_t n_interact = (size_t)interact_cnt;

    if (interact_coord == nullptr || interact_coord->n_rows != 3)
        throw std::invalid_argument("Input 'interact_coord' must have 3 rows.");

    if (interact_coord->n_cols != n_interact)
        throw std::invalid_argument("Number of columns of 'interact_coord' must match the sum of 'no_interact'.");

    constexpr dtype zero = dtype(0.0);
    constexpr dtype half = dtype(0.5);
    constexpr dtype los_limit = dtype(1.0e-4);

    // Set the output size
    if (path_length != nullptr && path_length->n_elem != n_path)
        path_length->set_size(n_path);
    if (fbs_pos != nullptr && (fbs_pos->n_rows != 3 || fbs_pos->n_cols != n_path))
        fbs_pos->set_size(3, n_path);
    if (lbs_pos != nullptr && (lbs_pos->n_rows != 3 || lbs_pos->n_cols != n_path))
        lbs_pos->set_size(3, n_path);
    if (path_angles != nullptr && (path_angles->n_rows != n_path || path_angles->n_cols != 4))
        path_angles->set_size(n_path, 4);
    if (path_coord != nullptr && path_coord->size() != n_path)
        path_coord->resize(n_path);

    // Get pointers
    const dtype *p_coord = interact_coord->memptr();
    dtype *p_length = path_length == nullptr ? nullptr : path_length->memptr();
    dtype *p_fbs = fbs_pos == nullptr ? nullptr : fbs_pos->memptr();
    dtype *p_lbs = lbs_pos == nullptr ? nullptr : lbs_pos->memptr();
    dtype *p_angles = path_angles == nullptr ? nullptr : path_angles->memptr();

    // Calculate half way point between TX and RX
    dtype TRx = Rx - Tx, TRy = Ry - Ty, TRz = Rz - Tz;
    TRx = Tx + half * TRx, TRy = Ty + half * TRy, TRz = Tz + half * TRz;

    for (size_t i_path = 0; i_path < n_path; ++i_path)
    {
        dtype fx = TRx, fy = TRy, fz = TRz;     // Initial FBS-Pos = half way point
        dtype lx = TRx, ly = TRy, lz = TRz;     // Initial LBS-Pos = half way point
        dtype x = Tx, y = Ty, z = Tz, d = zero; // Set segment start to TX position

        dtype *ppc = nullptr;
        if (path_coord != nullptr)
        {
            path_coord->at(i_path).set_size(3, p_interact[i_path] + 2);
            ppc = path_coord->at(i_path).memptr();
            *ppc++ = Tx, *ppc++ = Ty, *ppc++ = Tz;
        }

        // Get FBS and LBS positions
        for (unsigned ii = 0; ii < p_interact[i_path]; ++ii)
        {
            lx = *p_coord++, ly = *p_coord++, lz = *p_coord++;                      // Read segment end coordinate
            x -= lx, y -= ly, z -= lz;                                              // Calculate vector pointing from segment start to segment end
            d += std::sqrt(x * x + y * y + z * z);                                  // Add segment length to total path length
            x = lx, y = ly, z = lz;                                                 // Update segment start for next segment
            fx = ii == 0 ? lx : fx, fy = ii == 0 ? ly : fy, fz = ii == 0 ? lz : fz; // Sore FBS position (segment 0)
            if (ppc != nullptr)
                *ppc++ = lx, *ppc++ = ly, *ppc++ = lz;
        }
        x -= Rx, y -= Ry, z -= Rz;             // Calculate vector pointing last segment start to RX position
        d += std::sqrt(x * x + y * y + z * z); // Add last segment length to total path length

        if (p_length != nullptr)
            p_length[i_path] = d;

        if (p_fbs != nullptr)
        {
            if (reverse_path)
                p_fbs[3 * i_path] = lx, p_fbs[3 * i_path + 1] = ly, p_fbs[3 * i_path + 2] = lz;
            else
                p_fbs[3 * i_path] = fx, p_fbs[3 * i_path + 1] = fy, p_fbs[3 * i_path + 2] = fz;
        }

        if (p_lbs != nullptr)
        {
            if (reverse_path)
                p_lbs[3 * i_path] = fx, p_lbs[3 * i_path + 1] = fy, p_lbs[3 * i_path + 2] = fz;
            else
                p_lbs[3 * i_path] = lx, p_lbs[3 * i_path + 1] = ly, p_lbs[3 * i_path + 2] = lz;
        }

        if (p_angles != nullptr)
        {
            x = fx - Tx, y = fy - Ty, z = fz - Tz;
            d = std::sqrt(x * x + y * y + z * z);

            if (reverse_path)
            {
                p_angles[2 * n_path + i_path] = std::atan2(y, x);                        // AOD
                p_angles[3 * n_path + i_path] = d < los_limit ? zero : std::asin(z / d); // EOD
            }
            else
            {
                p_angles[i_path] = std::atan2(y, x);                                 // AOD
                p_angles[n_path + i_path] = d < los_limit ? zero : std::asin(z / d); // EOD
            }

            x = lx - Rx, y = ly - Ry, z = lz - Rz;
            d = std::sqrt(x * x + y * y + z * z);

            if (reverse_path)
            {
                p_angles[i_path] = std::atan2(y, x);                                 // AOA
                p_angles[n_path + i_path] = d < los_limit ? zero : std::asin(z / d); // EOA
            }
            else
            {
                p_angles[2 * n_path + i_path] = std::atan2(y, x);                        // AOA
                p_angles[3 * n_path + i_path] = d < los_limit ? zero : std::asin(z / d); // EOA
            }
        }
        if (ppc != nullptr)
            *ppc++ = Rx, *ppc++ = Ry, *ppc++ = Rz;

        if (reverse_path && path_coord != nullptr)
        {
            arma::uword n_elem = path_coord->at(i_path).n_elem;

            ppc = path_coord->at(i_path).memptr();
            dtype *tmp = new dtype[n_elem];
            std::memcpy(tmp, ppc, n_elem * sizeof(dtype));

            n_elem = path_coord->at(i_path).n_cols;
            for (arma::uword i_col = 0ULL; i_col < n_elem; ++i_col)
            {
                arma::uword ii = 3ULL * (n_elem - i_col - 1ULL);
                ppc[3 * i_col] = tmp[ii];
                ppc[3 * i_col + 1] = tmp[ii + 1];
                ppc[3 * i_col + 2] = tmp[ii + 2];
            }
            delete[] tmp;
        }
    }
}

template void quadriga_lib::coord2path(float Tx, float Ty, float Tz, float Rx, float Ry, float Rz, const arma::Col<unsigned> *no_interact, const arma::Mat<float> *interact_coord,
                                       arma::Col<float> *path_length, arma::Mat<float> *fbs_pos, arma::Mat<float> *lbs_pos, arma::Mat<float> *path_angles, std::vector<arma::Mat<float>> *path_coord, bool reverse_path);

template void quadriga_lib::coord2path(double Tx, double Ty, double Tz, double Rx, double Ry, double Rz, const arma::Col<unsigned> *no_interact, const arma::Mat<double> *interact_coord,
                                       arma::Col<double> *path_length, arma::Mat<double> *fbs_pos, arma::Mat<double> *lbs_pos, arma::Mat<double> *path_angles, std::vector<arma::Mat<double>> *path_coord, bool reverse_path);
