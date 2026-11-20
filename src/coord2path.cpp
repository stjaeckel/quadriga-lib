// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# coord2path
Convert path interaction coordinates into FBS/LBS positions, path length, and angles

- `n_interact` is a vector of length `n_path` with the number of interactions per path
- `interact_coord` stores all coordinates concatenated in path order, size `[3, sum(n_interact)]`
- LOS paths (`n_interact[i] == 0`) get a virtual FBS/LBS at the midpoint between TX and RX
- Set `reverse_path = true` to swap TX/RX and reverse all interaction sequences

## Declaration:
```
void quadriga_lib::coord2path(
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    const arma::u32_vec &n_interact,
    const arma::Mat<dtype> &interact_coord,
    arma::Col<dtype> *path_length = nullptr,
    arma::Mat<dtype> *fbs_pos = nullptr,
    arma::Mat<dtype> *lbs_pos = nullptr,
    arma::Mat<dtype> *path_angles = nullptr,
    std::vector<arma::Mat<dtype>> *path_coord = nullptr,
    bool reverse_path = false);
```

## Inputs:
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3, 1]` or `[1, 3]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3, 1]` or `[1, 3]`
- **`n_interact`** — Number of interactions per path (0 = LOS); `[n_path]`
- **`interact_coord`** — Interaction coordinates in path order; must have 3 rows; `[3, sum(n_interact)]`
- **`reverse_path`** *(optional)* — If `true`, swaps TX/RX and reverses interaction sequences

## Outputs:
- **`path_length`** *(optional)* — Absolute path length TX to RX; `[n_path]`
- **`fbs_pos`** *(optional)* — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** *(optional)* — Last-bounce scatterer positions; `[3, n_path]`
- **`path_angles`** *(optional)* — Departure and arrival angles {AOD, EOD, AOA, EOA}; `[n_path, 4]`
- **`path_coord`** *(optional)* — Full path coordinates including TX and RX; vector of `n_path` matrices, each `[3, n_interact+2]`
MD!*/

template <typename dtype>
void quadriga_lib::coord2path(const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &rx_pos,
                              const arma::u32_vec &n_interact, const arma::Mat<dtype> &interact_coord,
                              arma::Col<dtype> *path_length, arma::Mat<dtype> *fbs_pos, arma::Mat<dtype> *lbs_pos,
                              arma::Mat<dtype> *path_angles, std::vector<arma::Mat<dtype>> *path_coord, bool reverse_path)
{
    const dtype *p_tx = qd_xyz(tx_pos, "tx_pos"), *p_rx = qd_xyz(rx_pos, "rx_pos");
    const dtype Tx = p_tx[0], Ty = p_tx[1], Tz = p_tx[2], Rx = p_rx[0], Ry = p_rx[1], Rz = p_rx[2];

    size_t n_path = (size_t)n_interact.n_elem;

    // Calculate the total number of interactions (64-bit sum, a 32-bit sum can wrap)
    size_t n_interact_total = 0;
    const unsigned *p_interact = n_interact.memptr();
    for (size_t i = 0; i < n_path; ++i)
        n_interact_total += (size_t)p_interact[i];

    if (interact_coord.n_rows != 3)
        throw std::invalid_argument("Input 'interact_coord' must have 3 rows.");

    if (interact_coord.n_cols != n_interact_total)
        throw std::invalid_argument("Number of columns of 'interact_coord' must match the sum of 'n_interact'.");

    qd_check_alias(path_length, "path_length", &interact_coord);
    qd_check_alias(fbs_pos, "fbs_pos", &interact_coord);
    qd_check_alias(lbs_pos, "lbs_pos", &interact_coord);
    qd_check_alias(path_angles, "path_angles", &interact_coord);
    qd_check_alias_outputs(path_length, "path_length", fbs_pos, "fbs_pos");
    qd_check_alias_outputs(path_length, "path_length", lbs_pos, "lbs_pos");
    qd_check_alias_outputs(path_length, "path_length", path_angles, "path_angles");
    qd_check_alias_outputs(fbs_pos, "fbs_pos", lbs_pos, "lbs_pos");
    qd_check_alias_outputs(fbs_pos, "fbs_pos", path_angles, "path_angles");
    qd_check_alias_outputs(lbs_pos, "lbs_pos", path_angles, "path_angles");
    if (path_coord != nullptr)
        for (const auto &m : *path_coord)
            qd_check_alias(&m, "path_coord", &interact_coord);

    constexpr double los_limit = 1.0e-4;

    // Set the output size
    if (path_length != nullptr)
        path_length->set_size(n_path);
    if (fbs_pos != nullptr)
        fbs_pos->set_size(3, n_path);
    if (lbs_pos != nullptr)
        lbs_pos->set_size(3, n_path);
    if (path_angles != nullptr)
        path_angles->set_size(n_path, 4);
    if (path_coord != nullptr && path_coord->size() != n_path)
        path_coord->resize(n_path);

    // Get pointers
    const dtype *p_coord = interact_coord.memptr();
    dtype *p_length = path_length == nullptr ? nullptr : path_length->memptr();
    dtype *p_fbs = fbs_pos == nullptr ? nullptr : fbs_pos->memptr();
    dtype *p_lbs = lbs_pos == nullptr ? nullptr : lbs_pos->memptr();
    dtype *p_angles = path_angles == nullptr ? nullptr : path_angles->memptr();

    // Calculate half way point between TX and RX
    dtype TRx = std::fma((dtype)0.5, Tx, (dtype)0.5 * Rx);
    dtype TRy = std::fma((dtype)0.5, Ty, (dtype)0.5 * Ry);
    dtype TRz = std::fma((dtype)0.5, Tz, (dtype)0.5 * Rz);

    for (size_t i_path = 0; i_path < n_path; ++i_path)
    {
        dtype fx = TRx, fy = TRy, fz = TRz; // Initial FBS-Pos = half way point
        dtype lx = TRx, ly = TRy, lz = TRz; // Initial LBS-Pos = half way point

        // Set segment start to TX position
        double x = (double)Tx;
        double y = (double)Ty;
        double z = (double)Tz;
        double d = 0.0;

        dtype *ppc = nullptr;
        if (path_coord != nullptr)
        {
            path_coord->at(i_path).set_size(3, p_interact[i_path] + 2);
            ppc = path_coord->at(i_path).memptr();

            // Write Tx position to path coordinates
            *ppc++ = Tx;
            *ppc++ = Ty;
            *ppc++ = Tz;
        }

        // Get FBS and LBS positions
        for (unsigned ii = 0; ii < p_interact[i_path]; ++ii)
        {
            // Read segment end coordinate, store as LBS
            lx = *p_coord++;
            ly = *p_coord++;
            lz = *p_coord++;

            // Calculate vector pointing from segment start to segment end
            x -= (double)lx, y -= (double)ly, z -= (double)lz;

            // Add segment length to total path length
            d += std::sqrt(x * x + y * y + z * z);

            // Update segment start for next segment
            x = (double)lx, y = (double)ly, z = (double)lz;

            // Store FBS position (segment 0)
            fx = (ii == 0) ? lx : fx;
            fy = (ii == 0) ? ly : fy;
            fz = (ii == 0) ? lz : fz;

            // Write interaction location to path coordinates
            if (ppc != nullptr)
                *ppc++ = lx, *ppc++ = ly, *ppc++ = lz;
        }

        // Vector from the last segment start to the RX position
        x -= (double)Rx, y -= (double)Ry, z -= (double)Rz;

        // Add last segment length to total path length
        d += std::sqrt(x * x + y * y + z * z);

        // Store total path length
        if (p_length != nullptr)
            p_length[i_path] = (dtype)d;

        // Store total FBS position
        if (p_fbs != nullptr)
        {
            if (reverse_path)
                p_fbs[3 * i_path] = lx, p_fbs[3 * i_path + 1] = ly, p_fbs[3 * i_path + 2] = lz;
            else
                p_fbs[3 * i_path] = fx, p_fbs[3 * i_path + 1] = fy, p_fbs[3 * i_path + 2] = fz;
        }

        // Store total LBS position
        if (p_lbs != nullptr)
        {
            if (reverse_path)
                p_lbs[3 * i_path] = fx, p_lbs[3 * i_path + 1] = fy, p_lbs[3 * i_path + 2] = fz;
            else
                p_lbs[3 * i_path] = lx, p_lbs[3 * i_path + 1] = ly, p_lbs[3 * i_path + 2] = lz;
        }

        // Calculate angles
        if (p_angles != nullptr)
        {
            // Arrival angles; x, y, z already hold LBS - RX
            d = std::sqrt(x * x + y * y + z * z);

            if (reverse_path)
            {
                p_angles[i_path] = (dtype)std::atan2(y, x);                                         // AOD
                p_angles[n_path + i_path] = (d < los_limit) ? (dtype)0.0 : (dtype)std::asin(z / d); // EOD
            }
            else
            {
                p_angles[2 * n_path + i_path] = (dtype)std::atan2(y, x);                                // AOA
                p_angles[3 * n_path + i_path] = (d < los_limit) ? (dtype)0.0 : (dtype)std::asin(z / d); // EOA
            }

            // Departure angles
            x = (double)fx - (double)Tx, y = (double)fy - (double)Ty, z = (double)fz - (double)Tz;
            d = std::sqrt(x * x + y * y + z * z);

            if (reverse_path)
            {
                p_angles[2 * n_path + i_path] = (dtype)std::atan2(y, x);                                // AOA
                p_angles[3 * n_path + i_path] = (d < los_limit) ? (dtype)0.0 : (dtype)std::asin(z / d); // EOA
            }
            else
            {
                p_angles[i_path] = (dtype)std::atan2(y, x);                                         // AOD
                p_angles[n_path + i_path] = (d < los_limit) ? (dtype)0.0 : (dtype)std::asin(z / d); // EOD
            }
        }

        // Write Rx position to path coordinates
        if (ppc != nullptr)
            *ppc++ = Rx, *ppc++ = Ry, *ppc++ = Rz;

        // Reverse order in path_coord (if needed)
        if (reverse_path && path_coord != nullptr)
            path_coord->at(i_path) = arma::fliplr(path_coord->at(i_path));
    }
}

template void quadriga_lib::coord2path(const arma::Mat<float> &tx_pos, const arma::Mat<float> &rx_pos, const arma::u32_vec &n_interact, const arma::Mat<float> &interact_coord,
                                       arma::Col<float> *path_length, arma::Mat<float> *fbs_pos, arma::Mat<float> *lbs_pos, arma::Mat<float> *path_angles, std::vector<arma::Mat<float>> *path_coord, bool reverse_path);

template void quadriga_lib::coord2path(const arma::Mat<double> &tx_pos, const arma::Mat<double> &rx_pos, const arma::u32_vec &n_interact, const arma::Mat<double> &interact_coord,
                                       arma::Col<double> *path_length, arma::Mat<double> *fbs_pos, arma::Mat<double> *lbs_pos, arma::Mat<double> *path_angles, std::vector<arma::Mat<double>> *path_coord, bool reverse_path);
