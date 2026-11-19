// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Slot selection of the HDF5 read wrappers (qd_hdf5_select_slots): occupied slots of a selection, in column-major
// order, from the layout and channel IDs that hdf5_read_layout returns

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_helpers_wrappers.hpp"

#include <stdexcept>

TEST_CASE("qd_hdf5_select_slots - occupied slots of a selection in column-major order")
{
    // Layout 3 x 2 x 1 x 2; occupied: (2, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1)
    const arma::u32_vec dims = {3, 2, 1, 2};
    arma::u32_vec id(12, arma::fill::zeros);
    id(2) = 1, id(3) = 2, id(7) = 3;
    const arma::u32_vec none;

    const arma::u32_mat all = {{2, 0, 1}, {0, 1, 0}, {0, 0, 0}, {0, 0, 1}};
    CHECK(arma::all(arma::vectorise(qd_hdf5_select_slots(dims, id, none, none, none, none) == all)));

    // Given order and repeats are kept; empty slots are skipped; an empty vector is the full extent
    const arma::u32_vec ix = {2, 1, 2}, iy = {0};
    const arma::u32_mat expected = {{2, 2, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}};
    CHECK(arma::all(arma::vectorise(qd_hdf5_select_slots(dims, id, ix, iy, none, none) == expected)));
    const arma::u32_vec x1 = {1}, w0 = {0};
    const arma::u32_mat empty = qd_hdf5_select_slots(dims, id, x1, none, none, w0);
    CHECK(empty.n_rows == 4);
    CHECK(empty.n_cols == 0);

    // Indices outside the layout; a missing file (layout of zeros)
    const arma::u32_vec x3 = {0, 3}, w2 = {2};
    CHECK_THROWS_AS(qd_hdf5_select_slots(dims, id, x3, none, none, none), std::invalid_argument);
    CHECK_THROWS_AS(qd_hdf5_select_slots(dims, id, none, none, none, w2), std::invalid_argument);
    CHECK_THROWS_AS(qd_hdf5_select_slots(arma::u32_vec(4, arma::fill::zeros), arma::u32_vec(), none, none, none, none),
                    std::runtime_error);
}
