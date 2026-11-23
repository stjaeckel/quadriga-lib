// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// QDANT (XML) files: qdant_read of all entries, of a single ID with a layout, of a list with a duplicate, and of an ID that is not in the file

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &s)
{
    quadriga_lib::qdant_write(s.path("omni.qdant"), quadriga_lib::generate_antenna<double>("omni", 30.0));
    quadriga_lib::qdant_write(s.path("xpol.qdant"), quadriga_lib::generate_antenna<double>("xpol", 45.0));

    // Two IDs and a layout in one file
    std::string fn = s.path("two_ids.qdant");
    quadriga_lib::qdant_write(fn, quadriga_lib::generate_antenna<double>("3GPP", 30.0, 3.5e9, {}, {2, 1}, {}, {3}), 1);
    const arma::u32_mat layout = {{1, 2}};
    quadriga_lib::qdant_write(fn, quadriga_lib::generate_antenna<double>("half-wave-dipole", 45.0), 2, layout);
}

void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn = fz::write_file("in.qdant", data, size);

    fz::clean([&] { quadriga_lib::qdant_read<float>(fn); });
    fz::clean([&] { arma::u32_mat layout; quadriga_lib::qdant_read<double>(fn, {1}, &layout); });
    fz::clean([&] { quadriga_lib::qdant_read<float>(fn, {2, 1, 2}); });
    fz::clean([&] { quadriga_lib::qdant_read<double>(fn, {99}); });
}
