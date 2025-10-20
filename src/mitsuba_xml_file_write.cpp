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

#include "quadriga_lib.hpp"

#include <unordered_set>
#include <unordered_map>
#include <string_view>

#define PUGIXML_HEADER_ONLY
#include "pugixml.hpp"

// ----------------------------------------------------------------------------
// Helper: returns the ITU material that best matches `name`.
// Fallback is "vacuum".
// ----------------------------------------------------------------------------
static std::string_view map_to_itu(std::string_view name)
{
    // Extend this table once with all known mappings.
    static constexpr std::pair<std::string_view, std::string_view> kPrefixMap[] = {

        // Sionna default mappings
        {"concrete", "itu_concrete"},
        {"brick", "itu_brick"},
        {"plasterboard", "itu_plasterboard"},
        {"wood", "itu_wood"},
        {"glass", "itu_glass"},
        {"ceiling_board", "itu_ceiling_board"},
        {"chipboard", "itu_chipboard"},
        {"plywood", "itu_plywood"},
        {"marble", "itu_marble"},
        {"floorboard", "itu_floorboard"},
        {"metal", "itu_metal"},
        {"very_dry_ground", "itu_very_dry_ground"},
        {"medium_dry_ground", "itu_medium_dry_ground"},
        {"wet_ground", "itu_wet_ground"},

        // QRT additional materials
        {"textiles", "itu_wood"},
        {"plastic", "itu_wood"},
        {"ceramic", "itu_glass"},
        {"sea_water", "itu_wet_ground"},
        {"sea_ice", "itu_wet_ground"},
        {"water", "itu_wet_ground"},
        {"water_ice", "itu_wet_ground"},
        {"irr_glass", "itu_glass"},
    };

    // C++ 20 version:
    // for (auto [prefix, itu] : kPrefixMap)
    //     if (name.starts_with(prefix) || name.starts_with(itu))
    //         return itu;

    // C++ 17 version:
    for (auto [prefix, itu] : kPrefixMap)
        if ((name.size() >= prefix.size() && name.compare(0, prefix.size(), prefix) == 0) ||
            (name.size() >= itu.size() && name.compare(0, itu.size(), itu) == 0))
            return itu;

    return "vacuum";
}

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# mitsuba_xml_file_write
Write geometry and material data to a Mitsuba 3 XML scene file.

## Description:
This routine converts a triangular surface mesh stored in *quadriga-lib* data structures into the
XML format understood by **Mitsuba 3** <a href="https://www.mitsuba-renderer.org">www.mitsuba-renderer.org</a>.
The generated file can be loaded directly by **NVIDIA Sionna RT** for differentiable radio-propagation
simulations.<br><br>

- Converts a 3D geometry mesh into Mitsuba 3 XML format for use with rendering tools.
- Enables exporting models from `quadriga-lib` to be used with **Mitsuba 3** or **Sionna RT**:
- <a href="https://www.mitsuba-renderer.org">Mitsuba 3</a>: Research-oriented retargetable rendering system.
- <a href="https://developer.nvidia.com/sionna">NVIDIA Sionna</a>: Hardware-accelerated differentiable ray tracer for wireless propagation, built on Mitsuba 3.
- Supports grouping faces into named objects and assigning materials by name.
- Optionally maps materials to ITU default presets used by Sionna RT.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::mitsuba_xml_file_write(
                const std::string &fn,
                const arma::Mat<dtype> &vert_list,
                const arma::umat &face_ind,
                const arma::uvec &obj_ind,
                const arma::uvec &mtl_ind,
                const std::vector<std::string> &obj_names,
                const std::vector<std::string> &mtl_names,
                const arma::Mat<dtype> &bsdf = {},
                bool map_to_itu_materials = false);

```

## Arguments:
- `const std::string **fn**` (input)<br>
  Output file name (including path and `.xml` extension).

- `const arma::Mat<dtype> **vert_list**` (input)<br>
  Vertex list, size `[n_vert, 3]`, each row is a vertex (x, y, z) in Cartesian coordinates [m].

- `const arma::umat **face_ind**` (input)<br>
  Face indices (0-based), size `[n_mesh, 3]`, each row defines a triangle via vertex indices.

- `const arma::uvec **obj_ind**` (input)<br>
  Object indices (1-based), size `[n_mesh]`. Assigns each triangle to an object.

- `const arma::uvec **mtl_ind**` (input)<br>
  Material indices (1-based), size `[n_mesh]`. Assigns each triangle to a material.

- `const std::vector<std::string> **obj_names**` (input)<br>
  Names of objects. Length must be equal to `max(obj_ind)`.

- `const std::vector<std::string> **mtl_names**` (input)<br>
  Names of materials. Length must be equal to `max(mtl_ind)`.

- `const arma::Mat<dtype> **bsdf** = {}` (optional input)<br>
  Material reflectivity data (BSDF parameters), size `[mtl_names.size(), 17]`. If omitted, the `null` BSDF is used.
  Note that Sionna RT ignores all BSDF parameters. They are only used by the Mitsuma rendering system.
  See [[obj_file_read]] for a definition of the data fields.

- `bool **map_to_itu_materials** = false` (optional input)<br>
  If true, maps material names to ITU-defined presets used by Sionna RT. Default: `false`

## See also:
- [[obj_file_read]]
MD!*/

template <typename dtype>
void quadriga_lib::mitsuba_xml_file_write(const std::string &fn,
                                          const arma::Mat<dtype> &vert_list, const arma::umat &face_ind,
                                          const arma::uvec &obj_ind, const arma::uvec &mtl_ind,
                                          const std::vector<std::string> &obj_names,
                                          const std::vector<std::string> &mtl_names,
                                          const arma::Mat<dtype> &bsdf,
                                          bool map_to_itu_materials)
{
    if (face_ind.n_cols != 3)
        throw std::invalid_argument("Face indices 'face_ind' must have 3 columns (triangle mesh).");

    arma::uvec unique_obj_ind = arma::unique(obj_ind);
    arma::uword n_obj = arma::max(unique_obj_ind); // Number of mesh objects

    arma::uvec unique_mtl_ind = arma::unique(mtl_ind);
    arma::uword n_mtl = arma::max(unique_mtl_ind); // Number of mesh materials

    arma::uvec unique_face_ind = arma::unique(face_ind);
    arma::uword n_vert = arma::max(unique_face_ind) + 1; // Number of mesh vertices

    if (unique_obj_ind.n_elem != n_obj)
        throw std::invalid_argument("Found " + std::to_string(unique_obj_ind.n_elem) + " object indices, but expected " + std::to_string(n_obj) + ".");

    if (obj_names.size() != n_obj)
        throw std::invalid_argument("Found " + std::to_string(obj_names.size()) + " object names, but expected " + std::to_string(n_obj) + ".");

    if (unique_mtl_ind.n_elem != n_mtl)
        throw std::invalid_argument("Found " + std::to_string(unique_mtl_ind.n_elem) + " material indices, but expected " + std::to_string(n_mtl) + ".");

    if (mtl_names.size() != n_mtl)
        throw std::invalid_argument("Found " + std::to_string(mtl_names.size()) + " material names, but expected " + std::to_string(n_mtl) + ".");

    if (unique_face_ind.n_elem != n_vert)
        throw std::invalid_argument("Found " + std::to_string(unique_face_ind.n_elem) + " reference in 'face_ind', but expected " + std::to_string(n_vert) + ".");

    if (vert_list.n_rows != n_vert)
        throw std::invalid_argument("Found " + std::to_string(vert_list.n_rows) + " vertices, but expected " + std::to_string(n_vert) + ".");

    if (vert_list.n_cols != 3)
        throw std::invalid_argument("Vertex list mit have 3 columns.");

    // Fix invalid characters in obj names
    std::vector<std::string> obj_names_valid;
    obj_names_valid.reserve(obj_names.size());
    for (auto name : obj_names)
    {
        std::replace(name.begin(), name.end(), '/', '_');
        std::replace(name.begin(), name.end(), '\\', '_');
        obj_names_valid.push_back(name);
    }

    // Check for duplicate object names
    std::unordered_set<std::string> seen;
    seen.reserve(obj_names_valid.size());
    for (const auto &name : obj_names_valid)
    {
        auto [it, inserted] = seen.insert(name);
        if (!inserted)
            throw std::invalid_argument("Duplicate object name: '" + name + "'");
    }

    // Check for duplicate material names
    seen.clear();
    seen.reserve(mtl_names.size());
    for (const auto &name : mtl_names)
    {
        auto [it, inserted] = seen.insert(name);
        if (!inserted)
            throw std::invalid_argument("Duplicate material name: '" + name + "'");
    }

    bool use_bsdf = false;
    if (bsdf.n_rows != 0)
    {
        if (bsdf.n_rows != n_mtl)
            throw std::invalid_argument("Found " + std::to_string(bsdf.n_rows) + " BSDF entries, but expected " + std::to_string(n_mtl) + ".");
        if (bsdf.n_cols != 17)
            throw std::invalid_argument("BSDF Matrix mit have 17 columns.");
        use_bsdf = true;
    }

    // Map materials to ITU materials used by Sionna
    std::vector<std::string> mtl_names_local{mtl_names}; // copy for local edits
    arma::uvec mtl_ind_local{mtl_ind};                   // 1‒1 copy

    if (map_to_itu_materials)
    {
        use_bsdf = false;

        // Determine the unique ITU materials that will be needed
        std::unordered_map<std::string_view, arma::uword> itu_index; // ITU → 0-based index
        itu_index.reserve(16);

        for (arma::uword i_mtl = 0; i_mtl < n_mtl; ++i_mtl)
        {
            auto itu_name = map_to_itu(mtl_names[i_mtl]);

            // insert if not seen yet; value == next available 1-based index
            auto [it, inserted] = itu_index.emplace(itu_name, itu_index.size() + 1);

            // Remap every occurrence of the old material index to the ITU index
            const arma::uword old_idx = i_mtl + 1;  // original is 1-based
            const arma::uword new_idx = it->second; // also 1-based

            mtl_ind_local.elem(arma::find(mtl_ind == old_idx)).fill(new_idx);
        }

        // Build the name list in the order they were inserted
        mtl_names_local.clear();
        mtl_names_local.resize(itu_index.size());

        for (auto &&[itu_name, one_based_idx] : itu_index)
            mtl_names_local[one_based_idx - 1] = std::string{itu_name};
    }
    n_mtl = mtl_names_local.size();
    unique_mtl_ind = arma::unique(mtl_ind_local);

    if (n_mtl != arma::max(unique_mtl_ind))
        throw std::invalid_argument("Mismatch in material index.");

    // Check for duplicate material names
    seen.clear();
    seen.reserve(mtl_names_local.size());
    for (const auto &name : mtl_names_local)
    {
        auto [it, inserted] = seen.insert(name);
        if (!inserted)
            throw std::invalid_argument("Duplicate material name: '" + name + "'");
    }

    // Split objects with different materials assigned to the faces
    auto obj_names_local = obj_names_valid;
    auto obj_ind_local = obj_ind;
    obj_names_local.reserve(2 * n_obj);

    arma::uword next_obj_id = n_obj + 1;
    for (arma::uword i_obj = 1; i_obj <= n_obj; ++i_obj)
    {
        // Get the the original object name
        const std::string base_name = obj_names_valid[i_obj - 1];

        // Find all faces whose obj_ind == i_obj
        arma::uvec face_idx = arma::find(obj_ind == i_obj);

        //  Gather the material IDs for these faces
        arma::uvec mtls_here = mtl_ind_local.elem(face_idx);

        // Find the unique material IDs used by this object
        arma::uvec unique_mtls = arma::unique(mtls_here);
        if (unique_mtls.n_elem < 2)
            continue;

        // Rename the original object entry for the first material in unique_mtls
        {
            const arma::uword mtl_id = unique_mtls[0]; // 1-based
            std::string new_name = base_name + "_" + mtl_names_local[mtl_id - 1];
            obj_names_local[i_obj - 1] = new_name;
        }

        // For any additional materials, append new objects at the end of obj_names:
        for (arma::uword i_mtl = 1; i_mtl < unique_mtls.n_elem; ++i_mtl)
        {
            const arma::uword mtl_id = unique_mtls[i_mtl]; // 1-based
            std::string new_name = base_name + "_" + mtl_names_local[mtl_id - 1];
            obj_names_local.push_back(std::move(new_name));

            // Find all faces of this orig_obj that use mtl_id
            arma::uvec local_mask = arma::find(mtls_here == mtl_id);
            arma::uvec faces_for_this_new_object = face_idx.elem(local_mask);

            // Assign those faces to the new object‐ID
            for (arma::uword ii = 0; ii < faces_for_this_new_object.n_elem; ++ii)
                obj_ind_local[faces_for_this_new_object[ii]] = next_obj_id;
            ++next_obj_id;
        }
    }
    n_obj = next_obj_id - 1;

    // Check for duplicate object names
    seen.clear();
    seen.reserve(obj_names_local.size());
    for (const auto &name : obj_names_local)
    {
        auto [it, inserted] = seen.insert(name);
        if (!inserted)
            throw std::invalid_argument("Duplicate object name: '" + name + "'");
    }

    // Create folder for the mesh data
    std::filesystem::path xml_file(fn);
    std::filesystem::path mesh_folder_name = xml_file.stem().string() + "_meshes";
    std::filesystem::path mesh_folder = xml_file.parent_path() / mesh_folder_name;
    std::error_code ec;
    if (std::filesystem::exists(mesh_folder, ec))
    {
        if (!std::filesystem::is_directory(mesh_folder, ec))
            throw std::runtime_error("Cannot create folder '" + mesh_folder.string() + "': a file with that name already exists.");
    }
    else if (!std::filesystem::create_directory(mesh_folder, ec) || ec)
        throw std::runtime_error("Failed to create directory '" + mesh_folder.string() + "': " + ec.message());

    // Write PLY meshes
    for (arma::uword i_obj = 0; i_obj < n_obj; ++i_obj)
    {
        std::filesystem::path ply_file_name = obj_names_local[i_obj] + ".ply";
        std::filesystem::path ply_file = mesh_folder / ply_file_name;

        std::ofstream fileW(ply_file, std::ios::out | std::ios::binary);
        if (!fileW.is_open())
            throw std::runtime_error("Cannot open file '" + ply_file.string() + "'for writing.");

        std::string bin_data = "ply\nformat binary_little_endian 1.0\ncomment Created by Quadriga-Lib " + quadriga_lib::quadriga_lib_version() + "\n";
        fileW.write(bin_data.c_str(), bin_data.size());

        // Find all faces whose obj_ind == i_obj
        arma::uvec face_idx = arma::find(obj_ind_local == i_obj + 1);
        arma::uword n_faces = face_idx.n_elem;

        // Gather the vertex IDs for these faces
        arma::umat vert_idx = face_ind.rows(face_idx);
        arma::uvec unique_vert_idx = arma::unique(vert_idx);

        bin_data = "element vertex " + std::to_string(unique_vert_idx.n_elem) + "\n";
        fileW.write(bin_data.c_str(), bin_data.size());

        bin_data = "property float x\nproperty float y\nproperty float z\n";
        fileW.write(bin_data.c_str(), bin_data.size());

        bin_data = "element face " + std::to_string(n_faces) + "\n";
        fileW.write(bin_data.c_str(), bin_data.size());

        bin_data = "property list uchar uint vertex_indices\nend_header\n";
        fileW.write(bin_data.c_str(), bin_data.size());

        // Write vertices
        unsigned ply_face_idx = 0;
        arma::u32_mat ply_vert_idx(n_faces, 3);
        for (const auto &i_vert : unique_vert_idx)
        {
            if (i_vert > n_vert)
                throw std::runtime_error("Vertex list out of bound. This should not happen! Ever!");

            float x = (float)vert_list.at(i_vert, 0);
            float y = (float)vert_list.at(i_vert, 1);
            float z = (float)vert_list.at(i_vert, 2);

            fileW.write((char *)&x, sizeof(float));
            fileW.write((char *)&y, sizeof(float));
            fileW.write((char *)&z, sizeof(float));

            // Obtain indices relative to the local PLY file
            arma::uvec idx = arma::find(vert_idx == i_vert);
            ply_vert_idx.elem(idx).fill(ply_face_idx);
            ++ply_face_idx;
        }

        // Write faces
        unsigned char const_3 = (unsigned char)3;
        for (arma::uword i_face = 0; i_face < n_faces; ++i_face)
        {
            fileW.write((char *)&const_3, sizeof(unsigned char));
            for (arma::uword i = 0; i < 3; ++i)
            {
                unsigned e = ply_vert_idx.at(i_face, i);
                fileW.write((char *)&e, sizeof(unsigned));
            }
        }
        fileW.close();
    }

    // Write XML
    pugi::xml_document doc;
    auto node_scene = doc.append_child("scene");
    node_scene.append_attribute("version").set_value("3.0.0");

    // { // Add integrator
    //     auto node_integrator = node_scene.append_child("integrator");
    //     node_integrator.append_attribute("type").set_value("path");

    //     auto node_integer = node_integrator.append_child("integer");
    //     node_integer.append_attribute("name").set_value("max_depth");
    //     node_integer.append_attribute("value").set_value(5);
    // }

    // { // Add sensor
    //     auto node_sensor = node_scene.append_child("sensor");
    //     node_sensor.append_attribute("type").set_value("perspective");
    //     node_sensor.append_attribute("id").set_value("camera");
    //     node_sensor.append_attribute("name").set_value("camera");

    //     // fov axis
    //     auto node_string = node_sensor.append_child("string");
    //     node_string.append_attribute("name").set_value("fov_axis");
    //     node_string.append_attribute("value").set_value("x");

    //     // scalar floats
    //     auto node_float = node_sensor.append_child("float");
    //     node_float.append_attribute("name").set_value("fov");
    //     node_float.append_attribute("value").set_value(39.6);

    //     node_float = node_sensor.append_child("float");
    //     node_float.append_attribute("name").set_value("principal_point_offset_x");
    //     node_float.append_attribute("value").set_value(0.0);

    //     node_float = node_sensor.append_child("float");
    //     node_float.append_attribute("name").set_value("principal_point_offset_y");
    //     node_float.append_attribute("value").set_value(0.0);

    //     node_float = node_sensor.append_child("float");
    //     node_float.append_attribute("name").set_value("near_clip");
    //     node_float.append_attribute("value").set_value(0.1);

    //     node_float = node_sensor.append_child("float");
    //     node_float.append_attribute("name").set_value("far_clip");
    //     node_float.append_attribute("value").set_value(100.0);

    //     // to_world transform
    //     auto node_transform = node_sensor.append_child("transform");
    //     node_transform.append_attribute("name").set_value("to_world");

    //     auto node_rotate = node_transform.append_child("rotate");
    //     node_rotate.append_attribute("x").set_value(1);
    //     node_rotate.append_attribute("angle").set_value(116.44070980054053);

    //     node_rotate = node_transform.append_child("rotate");
    //     node_rotate.append_attribute("y").set_value(1);
    //     node_rotate.append_attribute("angle").set_value(5.419622430558914e-06);

    //     node_rotate = node_transform.append_child("rotate");
    //     node_rotate.append_attribute("z").set_value(1);
    //     node_rotate.append_attribute("angle").set_value(-133.30805320118378);

    //     auto node_translate = node_transform.append_child("translate");
    //     node_translate.append_attribute("value").set_value("7.358891 -6.925791 4.958309");

    //     // sampler
    //     auto node_sampler = node_sensor.append_child("sampler");
    //     node_sampler.append_attribute("type").set_value("independent");
    //     node_sampler.append_attribute("name").set_value("sampler");

    //     auto node_integer = node_sampler.append_child("integer");
    //     node_integer.append_attribute("name").set_value("sample_count");
    //     node_integer.append_attribute("value").set_value(16);

    //     // film
    //     auto node_film = node_sensor.append_child("film");
    //     node_film.append_attribute("type").set_value("hdrfilm");
    //     node_film.append_attribute("name").set_value("film");

    //     node_integer = node_film.append_child("integer");
    //     node_integer.append_attribute("name").set_value("width");
    //     node_integer.append_attribute("value").set_value(1920);

    //     node_integer = node_film.append_child("integer");
    //     node_integer.append_attribute("name").set_value("height");
    //     node_integer.append_attribute("value").set_value(1080);
    // }

    // { // Add emitter
    //     auto node_emitter = node_scene.append_child("emitter");
    //     node_emitter.append_attribute("type").set_value("point");
    //     node_emitter.append_attribute("id").set_value("elm__3");
    //     node_emitter.append_attribute("name").set_value("elm__3");

    //     // emitter position
    //     auto node_point = node_emitter.append_child("point");
    //     node_point.append_attribute("name").set_value("position");
    //     node_point.append_attribute("x").set_value(4.076245307922363);
    //     node_point.append_attribute("y").set_value(1.0054539442062378);
    //     node_point.append_attribute("z").set_value(5.903861999511719);

    //     // emitter intensity
    //     auto node_rgb = node_emitter.append_child("rgb");
    //     node_rgb.append_attribute("name").set_value("intensity");
    //     node_rgb.append_attribute("value").set_value("79.577469 79.577469 79.577469");
    // }

    // Write materials
    if (use_bsdf)
        for (arma::uword i_mtl = 0; i_mtl < n_mtl; ++i_mtl)
        {
            auto node_mtl = node_scene.append_child("bsdf");
            node_mtl.append_attribute("type").set_value("twosided");
            node_mtl.append_attribute("id").set_value(mtl_names_local[i_mtl].c_str());
            node_mtl.append_attribute("name").set_value(mtl_names_local[i_mtl].c_str());

            auto node_bsdf = node_mtl.append_child("bsdf");
            node_bsdf.append_attribute("type").set_value("principled");

            std::string rgb = std::to_string(bsdf(i_mtl, 0)) + "," + std::to_string(bsdf(i_mtl, 1)) + "," + std::to_string(bsdf(i_mtl, 2));
            auto node_rgb = node_bsdf.append_child("rgb");
            node_rgb.append_attribute("name").set_value("base_color");
            node_rgb.append_attribute("value").set_value(rgb.c_str());

            auto node_float = node_bsdf.append_child("float");
            node_float.append_attribute("name").set_value("metallic");
            node_float.append_attribute("value").set_value(bsdf(i_mtl, 5));

            node_float = node_bsdf.append_child("float");
            node_float.append_attribute("name").set_value("roughness");
            node_float.append_attribute("value").set_value(bsdf(i_mtl, 4));

            node_float = node_bsdf.append_child("float");
            node_float.append_attribute("name").set_value("anisotropic");
            node_float.append_attribute("value").set_value(bsdf(i_mtl, 14));

            node_float = node_bsdf.append_child("float");
            node_float.append_attribute("name").set_value("sheen");
            node_float.append_attribute("value").set_value(bsdf(i_mtl, 11));

            node_float = node_bsdf.append_child("float");
            node_float.append_attribute("name").set_value("clearcoat");
            node_float.append_attribute("value").set_value(bsdf(i_mtl, 12));

            node_float = node_bsdf.append_child("float");
            node_float.append_attribute("name").set_value("clearcoat_gloss");
            node_float.append_attribute("value").set_value((dtype)1.0 - bsdf(i_mtl, 13));

            node_float = node_bsdf.append_child("float");
            node_float.append_attribute("name").set_value("eta");
            node_float.append_attribute("value").set_value(bsdf(i_mtl, 6));
        }
    else
        for (arma::uword i_mtl = 0; i_mtl < n_mtl; ++i_mtl)
        {
            auto node_mtl = node_scene.append_child("bsdf");
            node_mtl.append_attribute("type").set_value("null");
            node_mtl.append_attribute("id").set_value(mtl_names_local[i_mtl].c_str());
            node_mtl.append_attribute("name").set_value(mtl_names_local[i_mtl].c_str());
        }

    // Write objects
    for (unsigned i_obj = 0; i_obj < n_obj; ++i_obj)
    {
        std::filesystem::path ply_file_name = obj_names_local[i_obj] + ".ply";
        std::filesystem::path ply_file = mesh_folder / ply_file_name;

        auto node_shape = node_scene.append_child("shape");
        node_shape.append_attribute("type").set_value("ply");
        node_shape.append_attribute("id").set_value(obj_names_local[i_obj].c_str());
        node_shape.append_attribute("name").set_value(obj_names_local[i_obj].c_str());

        auto node_string = node_shape.append_child("string");
        node_string.append_attribute("name").set_value("filename");
        node_string.append_attribute("value").set_value(ply_file.c_str());

        auto node_boolean = node_shape.append_child("boolean");
        node_boolean.append_attribute("name").set_value("face_normals");
        node_boolean.append_attribute("value").set_value("true");

        arma::uvec face_idx = arma::find(obj_ind_local == i_obj + 1, 1);
        arma::uword mtl_idx = mtl_ind_local.at(face_idx.at(0));

        if (mtl_idx - 1 > n_mtl)
            throw std::runtime_error("Material index out of bound. This should not happen! Ever!");

        auto node_ref = node_shape.append_child("ref");
        node_ref.append_attribute("id").set_value(mtl_names_local[mtl_idx - 1].c_str());
        node_ref.append_attribute("name").set_value("bsdf");
    }

    bool success = doc.save_file(xml_file.c_str(), "");
    if (!success)
        throw std::runtime_error("Cannot write file '" + xml_file.string() + "'.");
}

template void quadriga_lib::mitsuba_xml_file_write(const std::string &fn,
                                                   const arma::Mat<float> &vert_list, const arma::umat &face_ind,
                                                   const arma::uvec &obj_ind, const arma::uvec &mtl_ind,
                                                   const std::vector<std::string> &obj_names,
                                                   const std::vector<std::string> &mtl_names,
                                                   const arma::Mat<float> &bsdf,
                                                   bool map_to_itu_materials);

template void quadriga_lib::mitsuba_xml_file_write(const std::string &fn,
                                                   const arma::Mat<double> &vert_list, const arma::umat &face_ind,
                                                   const arma::uvec &obj_ind, const arma::uvec &mtl_ind,
                                                   const std::vector<std::string> &obj_names,
                                                   const std::vector<std::string> &mtl_names,
                                                   const arma::Mat<double> &bsdf,
                                                   bool map_to_itu_materials);