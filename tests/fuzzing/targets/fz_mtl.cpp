// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Wavefront MTL material files: obj_file_read of a fixed OBJ file whose "mtllib" is the fuzz input

#include "fuzz_target.hpp"

// Three faces with materials of the seed files and one face with a material that is in none of them
static const char *fixed_obj = "mtllib in.mtl\n"
                               "o Box\n"
                               "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                               "usemtl itu_brick\nf 1 2 3\n"
                               "usemtl itu_glass\nf 1 2 4\n"
                               "usemtl bla\nf 1 3 4\n"
                               "usemtl unknown_material\nf 2 3 4\n";

void fuzz_seeds(const fz::seeds &s)
{
    s.copy("tests/data/test_scene1.mtl");
    s.copy("tests/data/test_scene_pbr.mtl");
}

void fuzz_one(const uint8_t *data, size_t size)
{
    fz::write_file("in.mtl", data, size);
    std::string fn = fz::write_file("in.obj", fixed_obj);

    fz::clean([&]
              {
        arma::mat mesh, bsdf;
        arma::uvec mtl_ind;
        std::vector<std::string> mtl_names;
        quadriga_lib::obj_file_read<double>(fn, &mesh, nullptr, nullptr, nullptr, nullptr, &mtl_ind, &mtl_names, &bsdf); });
}
