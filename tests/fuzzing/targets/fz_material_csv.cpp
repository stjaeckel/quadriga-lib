// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Material tables (CSV) for the EM and acoustic material models: obj_file_read of a fixed OBJ file with the fuzz
// input as the material table, in the lenient and in the strict mode

#include "fuzz_target.hpp"

// Faces with materials of the RF table, the acoustic table and one that is in neither
static const char *fixed_obj = "o Room\n"
                               "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                               "usemtl itu_concrete\nf 1 2 3\n"
                               "usemtl concrete_150mm\nf 1 2 4\n"
                               "usemtl air\nf 1 3 4\n"
                               "usemtl unknown_material\nf 2 3 4\n";

void fuzz_seeds(const fz::seeds &s)
{
    s.copy("data/materials_rf.csv");
    s.copy("data/materials_acoustic.csv");
}

void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn_csv = fz::write_file("in.csv", data, size);
    std::string fn = fz::write_file("in.obj", fixed_obj);

    for (bool strict : {false, true})
        fz::clean([&]
                  {
            arma::mat mesh;
            arma::uvec csv_ind;
            std::vector<std::string> csv_names;
            std::unordered_map<std::string, std::vector<double>> csv_prop;
            quadriga_lib::obj_file_read<double>(fn, &mesh, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, fn_csv, &csv_ind, &csv_names, &csv_prop, strict); });
}
