// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// QDANT (XML) array antenna files: qdant_read with a layout, qdant_read of a second ID, qdant_read_multi

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &s)
{
    quadriga_lib::qdant_write(s.path("omni.qdant"), quadriga_lib::generate_arrayant_omni<double>(30.0));
    quadriga_lib::qdant_write(s.path("xpol.qdant"), quadriga_lib::generate_arrayant_xpol<double>(45.0));

    // Two IDs and a layout in one file
    std::string fn = s.path("two_ids.qdant");
    quadriga_lib::qdant_write(fn, quadriga_lib::generate_arrayant_3GPP<double>(2, 1, 3.5e9, 3, 0.0, 0.5, 1, 1, 0.5, 0.5, nullptr, 30.0), 1);
    const arma::u32_mat layout = {{1, 2}};
    quadriga_lib::qdant_write(fn, quadriga_lib::generate_arrayant_half_wave_dipole<double>(45.0), 2, &layout);
}

void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn = fz::write_file("in.qdant", data, size);

    fz::clean([&] { arma::u32_mat layout; quadriga_lib::qdant_read<double>(fn, 1, &layout); });
    fz::clean([&] { quadriga_lib::qdant_read<float>(fn, 2); });
    fz::clean([&] { quadriga_lib::qdant_read_multi<float>(fn); });
}
