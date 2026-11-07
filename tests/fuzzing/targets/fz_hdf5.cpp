// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// HDF5 channel files: hdf5_read_layout, hdf5_read_channel and the unstructured datasets (hdf5_read_dset_names,
// hdf5_read_dset) of the first slots

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &s)
{
    s.copy("tests/data/hdf5_v2_lossy.hdf5");

    // A 2 x 2 layout with two channels: one with structured data only, one with paths and unstructured data
    std::string fn = s.path("two_channels.hdf5");
    quadriga_lib::hdf5_create(fn, 2, 2);

    quadriga_lib::channel<double> c;
    c.name = "first";
    c.tx_pos = arma::mat(3, 1, arma::fill::ones);
    c.rx_pos = arma::mat(3, 1, arma::fill::value(5.0));
    c.coeff_re.push_back(arma::cube(1, 2, 2, arma::fill::value(0.5)));
    c.coeff_im.push_back(arma::cube(1, 2, 2, arma::fill::value(-0.5)));
    c.delay.push_back(arma::cube(1, 2, 2, arma::fill::value(1.0e-8)));
    quadriga_lib::hdf5_write(&c, fn, 0);

    c.name = "second";
    c.center_frequency = {3.5e9};
    c.path_gain.push_back(arma::vec({0.1, 0.01}));
    c.path_length.push_back(arma::vec({3.0, 7.5}));
    c.par_names = {"note", "gain", "grid"};
    c.par_data = {std::string("text"), 2.5, arma::mat(2, 3, arma::fill::value(1.25))};
    quadriga_lib::hdf5_write(&c, fn, 1, 1);
}

void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn = fz::write_file("in.hdf5", data, size);

    fz::clean([&] { arma::u32_vec channel_id; quadriga_lib::hdf5_read_layout(fn, &channel_id); });
    for (unsigned ix = 0; ix < 2; ++ix)
        for (unsigned iy = 0; iy < 2; ++iy)
        {
            fz::clean([&] { quadriga_lib::hdf5_read_channel<double>(fn, ix, iy); });
            std::vector<std::string> names;
            fz::clean([&] { quadriga_lib::hdf5_read_dset_names(fn, &names, ix, iy); });
            for (size_t i = 0; i < names.size() && i < 4; ++i)
                fz::clean([&] { quadriga_lib::hdf5_read_dset(fn, names[i], ix, iy); });
        }
}
