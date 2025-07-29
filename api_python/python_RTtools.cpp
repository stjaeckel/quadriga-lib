// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_icosphere.cpp"
#include "qpy_mitsuba_xml_file_write.cpp"
#include "qpy_obj_file_read.cpp"
#include "qpy_point_inside_mesh.cpp"
#include "qpy_triangle_mesh_segmentation.cpp"

void quadriga_lib_RTtools(py::module_ &m)
{
    m.def("icosphere", &icosphere, py::arg("n_div") = 1,
          py::arg("radius") = 1.0, py::arg("direction_xyz") = false);

    m.def("mitsuba_xml_file_write", &mitsuba_xml_file_write,
          py::arg("fn"),
          py::arg("vert_list") = py::array_t<double>(),
          py::arg("face_ind") = py::array_t<arma::uword>(),
          py::arg("obj_ind") = py::array_t<arma::uword>(),
          py::arg("mtl_ind") = py::array_t<arma::uword>(),
          py::arg("obj_names") = py::list(),
          py::arg("mtl_names") = py::list(),
          py::arg("bsdf") = py::array_t<double>(),
          py::arg("map_to_itu") = false);

    m.def("obj_file_read", &obj_file_read, py::arg("fn"));

    m.def("point_inside_mesh", &point_inside_mesh,
          py::arg("points") = py::array_t<double>(),
          py::arg("mesh") = py::array_t<double>(),
          py::arg("obj_ind") = py::array_t<unsigned>(),
          py::arg("distance") = 0.0);

    m.def("triangle_mesh_segmentation", &triangle_mesh_segmentation,
          py::arg("triangles") = py::array_t<double>(),
          py::arg("target_size") = 1024,
          py::arg("vec_size") = 1,
          py::arg("mtl_prop") = py::array_t<double>());
}
