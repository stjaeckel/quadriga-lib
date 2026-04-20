// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# combine_irs_coord
Combine path interaction coordinates for IRS-assisted TX → RX channels

- Merges two propagation segments (TX → IRS and IRS → RX) into complete path interaction coordinate sequences
- Interaction coordinates use a compressed format: `no_interact` counts interactions per path, `interact_coord` stores all coordinates sequentially in path order
- Each combined path appends segment 1 coordinates (optionally reversed) then the IRS position then segment 2 coordinates (optionally reversed); reversing affects coordinate order only, not endpoint positions
- Output contains at most `n_path_1 × n_path_2` paths; `active_path` (typically the return value of [[get_channels_irs]]) reduces this to active combinations only
- Typically used after [[get_channels_irs]] to produce interaction data for path visualization (e.g. in Blender) via [[coord2path]]

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

## Inputs:
- **`Ix, Iy, Iz`** — IRS position in Cartesian coordinates
- **`no_interact_1`** — Number of interaction points per path for segment 1 (TX → IRS); `[n_path_1]`
- **`interact_coord_1`** — Interaction coordinates for segment 1; `[3, sum(no_interact_1)]`
- **`no_interact_2`** — Number of interaction points per path for segment 2 (IRS → RX); `[n_path_2]`
- **`interact_coord_2`** — Interaction coordinates for segment 2; `[3, sum(no_interact_2)]`
- **`reverse_segment_1`** *(optional)* — If `true`, reverses interaction coordinate order within segment 1
- **`reverse_segment_2`** *(optional)* — If `true`, reverses interaction coordinate order within segment 2
- **`active_path`** *(optional)* — Boolean mask selecting path combinations to include; pass the return value of [[get_channels_irs]] directly; `[n_path_1 × n_path_2]`

## Outputs:
- **`no_interact`** — Number of interaction points per combined path; `[n_path_irs]`
- **`interact_coord`** — Combined interaction coordinates for all output paths; `[3, sum(no_interact)]`

## See also:
- [[get_channels_irs]] (generates `active_path` and channel coefficients for IRS channels)
- [[coord2path]] (consumes interaction coordinates to compute angles and path geometry)
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
            arma::uword n_int_2 = (arma::uword)pI2[i_path_2];

            if (!get_subset || active_path->at(i_path))
            {
                arma::Mat<dtype> seg_2;
                if (n_int_2 != 0ULL)
                    seg_2 = interact_coord_2->cols(i_start_2, i_start_2 + n_int_2 - 1ULL);
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

            i_start_2 += n_int_2;
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
