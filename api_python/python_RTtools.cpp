// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_calc_diffraction_gain.cpp"
#include "qpy_cube.cpp"
#include "qpy_generate_diffraction_paths.cpp"
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
#include "qpy_plane.cpp"
#include "qpy_subdivide_triangles.cpp"

void quadriga_lib_RTtools(py::module_ &m)
{
    m.def("calc_diffraction_gain", &calc_diffraction_gain,
          py::arg("orig").none(false),
          py::arg("dest").none(false),
          py::arg("mesh").none(false),
          py::arg("mtl_ind").none(false),
          py::arg("mtl_prop").none(false),
          py::arg("center_freq"),
          py::kw_only(),
          py::arg("lod") = 2,
          py::arg("verbose") = 0,
          py::arg("sub_mesh_ind") = py::none(),
          py::arg("use_kernel") = 0,
          py::arg("gpu_id") = 0,
          py::arg("scalar_mode") = false,
          py::arg("thin_slab_threshold") = 0.0,
          py::arg("apply_fspl") = false,
          py::arg("apply_phase") = false);

    m.def("cube", &cube,
          py::arg("scale") = py::none(),
          py::arg("rotation") = py::none(),
          py::arg("location") = py::none(),
          py::kw_only(),
          py::arg("n_div") = 1);

    m.def("icosphere", &icosphere, py::arg("n_div"),
          py::arg("radius"), py::kw_only(), py::arg("direction_xyz") = false);

    m.def("mitsuba_xml_file_write", &mitsuba_xml_file_write,
          py::arg("fn"),
          py::arg("vert_list").none(false),
          py::arg("face_ind").none(false),
          py::arg("obj_ind").none(false),
          py::arg("mtl_ind").none(false),
          py::arg("obj_names").none(false),
          py::arg("mtl_names").none(false),
          py::arg("bsdf") = py::none(),
          py::kw_only(),
          py::arg("map_to_itu_materials") = false);

    m.def("obj_file_read", &obj_file_read,
          py::arg("fn"),
          py::arg("fn_csv") = "",
          py::kw_only(),
          py::arg("csv_strict") = false);

    m.def("obj_file_write", &obj_file_write,
          py::arg("fn"),
          py::arg("mesh") = py::none(),
          py::arg("obj_ind") = py::none(),
          py::arg("mtl_ind") = py::none(),
          py::arg("obj_names") = py::none(),
          py::arg("mtl_names") = py::none(),
          py::kw_only(),
          py::arg("vert_list") = py::none(),
          py::arg("face_ind") = py::none(),
          py::arg("bsdf") = py::none(),
          py::arg("threshold") = 0.001,
          py::arg("csv_ind") = py::none(),
          py::arg("csv_names") = py::none(),
          py::arg("csv_prop") = py::none(),
          py::arg("csv_write_defaults") = false,
          py::arg("split_loose_parts") = false);

    m.def("plane", &plane,
          py::arg("scale") = py::none(),
          py::arg("rotation") = py::none(),
          py::arg("location") = py::none(),
          py::kw_only(),
          py::arg("n_div") = 1);

    m.def("point_cloud_aabb", &point_cloud_aabb,
          py::arg("points").none(false),
          py::arg("sub_cloud_ind") = py::none(),
          py::kw_only(),
          py::arg("vec_size") = 1);

    m.def("point_cloud_segmentation", &point_cloud_segmentation,
          py::arg("points").none(false),
          py::kw_only(),
          py::arg("target_size") = 1024,
          py::arg("vec_size") = 1);

    m.def("point_inside_mesh", &point_inside_mesh,
          py::arg("points").none(false),
          py::arg("mesh").none(false),
          py::arg("obj_ind") = py::none(),
          py::kw_only(),
          py::arg("distance") = 0.0);

    m.def("ray_point_intersect", &ray_point_intersect,
          py::arg("points").none(false),
          py::arg("orig").none(false),
          py::arg("trivec").none(false),
          py::arg("tridir").none(false),
          py::arg("sub_cloud_ind") = py::none(),
          py::kw_only(),
          py::arg("use_kernel") = 0,
          py::arg("gpu_id") = 0);

    m.def("ray_triangle_intersect", &ray_triangle_intersect,
          py::arg("orig").none(false),
          py::arg("dest").none(false),
          py::arg("mesh").none(false),
          py::arg("sub_mesh_ind") = py::none(),
          py::arg("aabb") = py::none(),
          py::kw_only(),
          py::arg("use_kernel") = 0,
          py::arg("gpu_id") = 0);

    m.def("triangle_mesh_aabb", &triangle_mesh_aabb,
          py::arg("mesh").none(false),
          py::arg("sub_mesh_ind") = py::none(),
          py::kw_only(),
          py::arg("vec_size") = 1);

    m.def("triangle_mesh_segmentation", &triangle_mesh_segmentation,
          py::arg("mesh").none(false),
          py::kw_only(),
          py::arg("target_size") = 1024,
          py::arg("vec_size") = 1,
          py::arg("mtl_ind") = py::none());

    m.def("generate_diffraction_paths", &generate_diffraction_paths,
          py::arg("orig").none(false),
          py::arg("dest").none(false),
          py::arg("center_freq"),
          py::arg("lod"),
          py::kw_only(),
          py::arg("scalar_mode") = false);

    m.def("subdivide_triangles", &subdivide_triangles,
          py::arg("n_div"),
          py::arg("triangles_in").none(false),
          py::arg("mtl_ind") = py::none());
}
