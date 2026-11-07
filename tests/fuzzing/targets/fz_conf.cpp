// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Scenario parameter files (.conf): conf_file_read with all outputs, with and without the value checks

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &s)
{
    for (const auto &entry : std::filesystem::directory_iterator(std::string(QD_SOURCE_DIR) + "/data/config"))
        if (entry.path().extension() == ".conf")
            s.copy("data/config/" + entry.path().filename().string());
}

void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn = fz::write_file("in.conf", data, size);

    for (bool validate : {false, true})
        fz::clean([&]
                  {
            std::unordered_map<std::string, double> scenpar;
            std::string pl_model;
            std::unordered_map<std::string, std::vector<double>> plpar;
            arma::mat lsp_xcorr;
            std::vector<std::string> key_order, pl_order, short_names, ignored_keys;
            quadriga_lib::conf_file_read(fn, scenpar, &pl_model, &plpar, &lsp_xcorr, &key_order, &pl_order,
                                         &short_names, &ignored_keys, validate); });
}
