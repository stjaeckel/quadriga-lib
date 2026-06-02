// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_icosphere.cpp"
#include "qpy_mitsuba_xml_file_write.cpp"
#include "qpy_obj_file_read.cpp"
#include "qpy_obj_file_write.cpp"
#include "qpy_point_cloud_aabb.cpp"
#include "qpy_point_cloud_segmentation.cpp"
#include "qpy_point_inside_mesh.cpp"
#include "qpy_ray_point_intersect.cpp"
#include "qpy_ray_triangle_intersect.cpp"
#include "qpy_triangle_mesh_aabb.cpp"
#include "qpy_triangle_mesh_segmentation.cpp"
#include "qpy_calc_diffraction_gain.cpp"

void quadriga_lib_RTtools(py::module_ &m)
{
    m.def("calc_diffraction_gain", &calc_diffraction_gain,
          py::arg("orig"),
          py::arg("dest"),
          py::arg("mesh"),
          py::arg("mtl_ind"),
          py::arg("mtl_prop"),
          py::arg("center_frequency"),
          py::arg("lod") = 2,
          py::arg("verbose") = 0,
          py::arg("sub_mesh_index") = py::none(),
          py::arg("use_kernel") = 0,
          py::arg("gpu_id") = 0,
          py::arg("scalar_mode") = false);

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

    m.def("obj_file_read", &obj_file_read,
          py::arg("fn") = std::string(""),
          py::arg("fn_csv") = std::string(""),
          py::arg("csv_strict") = false);

    m.def("obj_file_write", &obj_file_write,
          py::arg("fn") = "",
          py::arg("mesh") = py::none(),
          py::arg("obj_ind") = py::none(),
          py::arg("mtl_ind") = py::none(),
          py::arg("obj_names") = py::none(),
          py::arg("mtl_names") = py::none(),
          py::arg("vert_list") = py::none(),
          py::arg("face_ind") = py::none(),
          py::arg("bsdf") = py::none(),
          py::arg("threshold") = 0.001);

    m.def("point_cloud_aabb", &point_cloud_aabb,
          py::arg("points") = py::array_t<double>(),
          py::arg("sub_cloud_ind") = py::array_t<unsigned>(),
          py::arg("vec_size") = 1);

    m.def("point_cloud_segmentation", &point_cloud_segmentation,
          py::arg("points") = py::array_t<double>(),
          py::arg("target_size") = 1024,
          py::arg("vec_size") = 1);

    m.def("point_inside_mesh", &point_inside_mesh,
          py::arg("points"),
          py::arg("mesh"),
          py::arg("obj_ind") = py::none(),
          py::arg("distance") = 0.0);

    m.def("ray_point_intersect", &ray_point_intersect,
          py::arg("orig") = py::array_t<double>(),
          py::arg("trivec") = py::array_t<double>(),
          py::arg("tridir") = py::array_t<double>(),
          py::arg("points") = py::array_t<double>(),
          py::arg("sub_cloud_ind") = py::array_t<unsigned>(),
          py::arg("use_kernel") = 0,
          py::arg("gpu_id") = 0);

    m.def("ray_triangle_intersect", &ray_triangle_intersect,
          py::arg("orig") = py::array_t<double>(),
          py::arg("dest") = py::array_t<double>(),
          py::arg("mesh") = py::array_t<double>(),
          py::arg("sub_mesh_index") = py::array_t<unsigned>(),
          py::arg("aabb") = py::array_t<double>(),
          py::arg("use_kernel") = 0,
          py::arg("gpu_id") = 0);

    m.def("triangle_mesh_aabb", &triangle_mesh_aabb,
          py::arg("triangles") = py::array_t<double>(),
          py::arg("sub_mesh_index") = py::array_t<unsigned>(),
          py::arg("vec_size") = 1);

    m.def("triangle_mesh_segmentation", &triangle_mesh_segmentation,
          py::arg("triangles"),
          py::arg("target_size") = 1024,
          py::arg("vec_size") = 1,
          py::arg("mtl_ind") = py::none());
}
