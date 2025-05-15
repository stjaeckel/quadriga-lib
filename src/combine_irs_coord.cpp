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

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# combine_irs_coord
Combine path interaction coordinates for channels with intelligent reflective surfaces (IRS)

## Description:
- Merges two propagation segments — (1) TX → IRS and (2) IRS → RX — into complete TX → RX paths via an IRS.
- Interaction coordinates of both segments are stored in a compressed format where `no_interact` is a vector of
  length `n_path` storing the number of interactions per path and `interact_coord` stores all
  interaction coordinates in path order.
- Each combined path includes interaction points from both segments, optionally reversed for each segment.
- The number of output paths `n_path_irs` is at most `n_path_1 × n_path_2`, but can be reduced by specifying `active_path`.
- Output includes the number and coordinates of interaction points per IRS-reflected path.
- Allowed datatypes (`dtype`): `float` or `double`.

## Declaration:
```
void quadriga_lib::combine_irs_coord(
                dtype Ix, dtype Iy, dtype Iz,
                const arma::u32_vec *no_interact_1,
                const arma::Mat<dtype> *interact_coord_1,
                const arma::u32_vec *no_interact_2,
                const arma::Mat<dtype> *interact_coord_2,
                arma::u32_vec *no_interact,
                arma::Mat<dtype> *interact_coord,
                bool reverse_segment_1 = false,
                bool reverse_segment_2 = false,
                const std::vector<bool> *active_path = nullptr);
```

## Arguments:
- `dtype **Ix**, **Iy**, **Iz**` (input)<br>
  IRS position in Cartesian coordinates `[m]`.

- `const arma::u32_vec ***no_interact_1**` (input)<br>
  Number of interaction points in segment 1 (TX → IRS), vector of length `[n_path_1]`

- `const arma::Mat<dtype> ***interact_coord_1**` (input)<br>
  Interaction coordinates for segment 1, matrix of size `[3, sum(no_interact_1)]`.

- `const arma::u32_vec ***no_interact_2**` (input)<br>
  Number of interaction points in segment 2 (IRS → RX), vector of length `[n_path_2]`

- `const arma::Mat<dtype> ***interact_coord_2**` (input)<br>
  Interaction coordinates for segment 2, matrix of size `[3, sum(no_interact_2)]`.

- `arma::u32_vec ***no_interact**` (output)<br>
  Combined number of interaction points per IRS-based path, vector of length `[n_path_irs]`.

- `arma::Mat<dtype> ***interact_coord**` (output)<br>
  Combined interaction coordinates, matrix of size `[3, sum(no_interact)]`.

- `bool **reverse_segment_1** = false` (optional input)<br>
  If `true`, reverses the interaction coordinates of segment 1. TX and IRS positions are not swapped. Default: `false`.

- `bool **reverse_segment_2** = false` (optional input)<br>
  If `true`, reverses the interaction coordinates of segment 2. IRS and RX positions are not swapped. Default: `false`.

- `const std::vector<bool> ***active_path** = nullptr` (optional input)<br>
  Optional mask vector of length `[n_path_1 × n_path_2]`. If provided, only paths with `true` are combined. 
  This is generated as the output of [[get_channels_irs]]

## Technical Notes:
- Paths are created by appending interaction coordinates from both segments. Reversing only affects the order of these coordinates.
- Output matrix `interact_coord` is built sequentially and should be preallocated only if the total number of interaction points is known in advance.
- The function supports sparsely activated paths via `active_path`, this is generated as a return value of [[get_channels_irs]]
  and filters out paths that have very little power after being reflected by the IRS. 
- This function is typically used as a complementary step to delay or coefficient calculations involving IRS-based channels. 
  It calculated the required data for visualizing paths, e.g. in Blender or other visualization tools.

## See also:
- [[get_channels_irs]] (for computing IRS channels)
- [[coord2path]] (for processing coordinates, calculating departure and arrival angels, etc.)
MD!*/

// Combine path interaction coordinates for Intelligent Reflective Surfaces (IRS)
template <typename dtype>
void quadriga_lib::combine_irs_coord(dtype Ix, dtype Iy, dtype Iz,
                                     const arma::u32_vec *no_interact_1, const arma::Mat<dtype> *interact_coord_1,
                                     const arma::u32_vec *no_interact_2, const arma::Mat<dtype> *interact_coord_2,
                                     arma::u32_vec *no_interact, arma::Mat<dtype> *interact_coord,
                                     bool reverse_segment_1, bool reverse_segment_2, const std::vector<bool> *active_path)
{
    if (no_interact_1 == nullptr)
        throw std::invalid_argument("Input 'no_interact_1' cannot be NULL.");
    if (interact_coord_1 == nullptr)
        throw std::invalid_argument("Input 'interact_coord_1' cannot be NULL.");
    if (no_interact_2 == nullptr)
        throw std::invalid_argument("Input 'no_interact_2' cannot be NULL.");
    if (interact_coord_2 == nullptr)
        throw std::invalid_argument("Input 'interact_coord_2' cannot be NULL.");
    if (no_interact == nullptr)
        throw std::invalid_argument("Output 'no_interact' cannot be NULL.");
    if (interact_coord == nullptr)
        throw std::invalid_argument("Output 'interact_coord' cannot be NULL.");

    arma::uword n_path_1 = no_interact_1->n_elem;
    arma::uword n_path_2 = no_interact_2->n_elem;
    arma::uword n_path = n_path_1 * n_path_2;
    arma::uword n_path_irs = n_path;

    bool get_subset = active_path != nullptr;
    if (get_subset)
    {
        if (active_path->size() != n_path)
            throw std::invalid_argument("Number of entries in 'active_path' must match n_path_1 * n_path_2.");

        n_path_irs = 0ULL;
        for (bool active : *active_path)
            if (active)
                ++n_path_irs;

        get_subset = n_path_irs != n_path;
    }

    if (n_path_irs == 0ULL)
    {
        no_interact->reset();
        interact_coord->reset();
        return;
    }

    // Get the number of interactions in the output
    no_interact->set_size(n_path_irs);
    const unsigned *pI1 = no_interact_1->memptr();
    const unsigned *pI2 = no_interact_2->memptr();
    unsigned *pI = no_interact->memptr();

    arma::uword i_path_irs = 0ULL, i_path = 0ULL, sum_no_interact = 0ULL;
    for (arma::uword i_path_1 = 0ULL; i_path_1 < n_path_1; ++i_path_1)
    {
        unsigned i1 = pI1[i_path_1] + 1U;
        for (arma::uword i_path_2 = 0ULL; i_path_2 < n_path_2; ++i_path_2)
        {
            if (!get_subset || active_path->at(i_path))
            {
                unsigned i = i1 + pI2[i_path_2];
                pI[i_path_irs] = i;
                sum_no_interact += (arma::uword)i;
                ++i_path_irs;
            }
            ++i_path;
        }
    }

    // Calculate the output coordinates
    i_path_irs = 0ULL, i_path = 0ULL;
    interact_coord->set_size(3ULL, sum_no_interact);

    arma::uword i_start_1 = 0ULL, i_col = 0ULL, n_bytes_per_col = 3ULL * sizeof(dtype);
    for (arma::uword i_path_1 = 0ULL; i_path_1 < n_path_1; ++i_path_1)
    {
        arma::Mat<dtype> seg_1;
        arma::uword n_int_1 = (arma::uword)pI1[i_path_1];

        if (n_int_1 != 0ULL)
        {
            arma::uword i_end_1 = i_start_1 + n_int_1 - 1ULL;
            seg_1 = interact_coord_1->cols(i_start_1, i_end_1);
            i_start_1 = i_end_1 + 1ULL;
        }
        if (reverse_segment_1 && n_int_1 > 1ULL)
            seg_1 = arma::fliplr(seg_1);

        arma::uword i_start_2 = 0ULL;
        for (arma::uword i_path_2 = 0ULL; i_path_2 < n_path_2; ++i_path_2)
        {
            if (!get_subset || active_path->at(i_path))
            {
                arma::Mat<dtype> seg_2;
                arma::uword n_int_2 = (arma::uword)pI2[i_path_2];

                if (n_int_2 != 0ULL)
                {
                    arma::uword i_end_2 = i_start_2 + n_int_2 - 1ULL;
                    seg_2 = interact_coord_2->cols(i_start_2, i_end_2);
                    i_start_2 = i_end_2 + 1ULL;
                }
                if (reverse_segment_2 && n_int_2 > 1ULL)
                    seg_2 = arma::fliplr(seg_2);

                if (n_int_1 != 0ULL)
                {
                    std::memcpy(interact_coord->colptr(i_col), seg_1.memptr(), n_int_1 * n_bytes_per_col);
                    i_col += n_int_1;
                }

                dtype *p_col = interact_coord->colptr(i_col);
                p_col[0] = Ix;
                p_col[1] = Iy;
                p_col[2] = Iz;
                ++i_col;

                if (n_int_2 != 0ULL)
                {
                    std::memcpy(interact_coord->colptr(i_col), seg_2.memptr(), n_int_2 * n_bytes_per_col);
                    i_col += n_int_2;
                }
            }
            ++i_path;
        }
    }
}

template void quadriga_lib::combine_irs_coord(float Ix, float Iy, float Iz,
                                              const arma::u32_vec *no_interact_1, const arma::Mat<float> *interact_coord_1,
                                              const arma::u32_vec *no_interact_2, const arma::Mat<float> *interact_coord_2,
                                              arma::u32_vec *no_interact, arma::Mat<float> *interact_coord,
                                              bool reverse_segment_1, bool reverse_segment_2, const std::vector<bool> *active_path);

template void quadriga_lib::combine_irs_coord(double Ix, double Iy, double Iz,
                                              const arma::u32_vec *no_interact_1, const arma::Mat<double> *interact_coord_1,
                                              const arma::u32_vec *no_interact_2, const arma::Mat<double> *interact_coord_2,
                                              arma::u32_vec *no_interact, arma::Mat<double> *interact_coord,
                                              bool reverse_segment_1, bool reverse_segment_2, const std::vector<bool> *active_path);
