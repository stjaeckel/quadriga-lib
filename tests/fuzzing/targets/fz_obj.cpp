// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Wavefront OBJ files: obj_file_read with all mesh, vertex, face, object and material outputs

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &s)
{
    s.copy("tests/data/test_scene.obj");
    s.copy("tests/data/test_scene_pbr.obj");

    // Two objects, two materials, a shared vertex
    arma::mat mesh = {{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                      {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}};
    arma::uvec obj_ind = {0, 1}, mtl_ind = {1, 2};
    std::vector<std::string> obj_names = {"Wall", "Floor"}, mtl_names = {"itu_concrete", "itu_glass"};
    quadriga_lib::obj_file_write<double>(s.path("two_objects.obj"), &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names);
}

void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn = fz::write_file("in.obj", data, size);

    fz::clean([&]
              {
        arma::mat mesh, vert_list, bsdf;
        arma::umat face_ind;
        arma::uvec obj_ind, mtl_ind;
        std::vector<std::string> obj_names, mtl_names;
        quadriga_lib::obj_file_read<double>(fn, &mesh, &vert_list, &face_ind, &obj_ind, &obj_names, &mtl_ind,
                                            &mtl_names, &bsdf); });
}
