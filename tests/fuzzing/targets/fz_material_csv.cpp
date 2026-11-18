// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Material tables (CSV) for the EM and acoustic material models: obj_file_read of a fixed OBJ file with the fuzz
// input as the material table, in the lenient and in the strict mode; every material of a table that reads is then
// evaluated by medium_gain, interface_gain and refractive_index, in EM and in scalar mode

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
                                                nullptr, fn_csv, &csv_ind, &csv_names, &csv_prop, strict);
            for (arma::uword iM = 1; iM <= (arma::uword)csv_names.size(); ++iM)
                for (bool scalar_mode : {false, true})
                {
                    const double f = scalar_mode ? 1000.0 : 3.5e9; // 1 kHz acoustic, 3.5 GHz EM
                    quadriga_lib::medium_gain<double>(csv_prop, iM, 0.1, f, scalar_mode);
                    quadriga_lib::interface_gain<double>(csv_prop, iM, f, scalar_mode);
                    quadriga_lib::refractive_index<double>(csv_prop, iM, f, scalar_mode);
                } });
}
