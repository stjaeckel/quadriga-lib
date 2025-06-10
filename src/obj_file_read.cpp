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

#include "quadriga_tools.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# obj_file_read
Read Wavefront `.obj` file and extract geometry and material information

## Description:
- Parses a Wavefront `.obj` file containing triangularized 3D geometry.
- Extracts triangle face data, material properties, vertex indices, and optional metadata such as object/material names.
- Multiple triangles belonging to the same object are grouped together by `obj_ind`.
- Supports default and custom ITU-compliant materials encoded via the `usemtl` tag.
- Automatically resizes output matrices/vectors as needed to match the file content.
- Returns the number of triangular mesh elements found in the file.

- Allowed datatypes (`dtype`): `float` or `double`.

## Declaration:
```
arma::uword quadriga_lib::obj_file_read(
                std::string fn,
                arma::Mat<dtype> *mesh = nullptr,
                arma::Mat<dtype> *mtl_prop = nullptr,
                arma::Mat<dtype> *vert_list = nullptr,
                arma::umat *face_ind = nullptr,
                arma::uvec *obj_ind = nullptr,
                arma::uvec *mtl_ind = nullptr,
                std::vector<std::string> *obj_names = nullptr,
                std::vector<std::string> *mtl_names = nullptr,
                arma::Mat<dtype> *bsdf = nullptr);
```

## Arguments:
- `std::string **fn**` (input)<br>
  Path to the `.obj` file to be read.

- `arma::Mat<dtype> ***mesh** = nullptr` (optional output)<br>
  Flattened triangle mesh data. Each row holds `[X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3]`. Size: `[n_mesh, 9]`.

- `arma::Mat<dtype> ***mtl_prop** = nullptr` (optional output)<br>
  Material properties for each triangle. Size: `[n_mesh, 5]`.

- `arma::Mat<dtype> ***vert_list** = nullptr` (optional output)<br>
  List of all vertex positions found in the `.obj` file. Size: `[n_vert, 3]`.

- `arma::umat ***face_ind** = nullptr` (optional output)<br>
  Indices into `vert_list` for each triangle (0-based). Size: `[n_mesh, 3]`.

- `arma::uvec ***obj_ind** = nullptr` (optional output)<br>
  Object index (1-based) for each triangle. Size: `[n_mesh]`.

- `arma::uvec ***mtl_ind** = nullptr` (optional output)<br>
  Material index (1-based) for each triangle. Size: `[n_mesh]`.

- `std::vector<std::string> ***obj_names** = nullptr` (optional output)<br>
  Names of objects found in the file. Length: `max(obj_ind)`.

- `std::vector<std::string> ***mtl_names** = nullptr` (optional output)<br>
  Names of materials found in the file. Length: `max(mtl_ind)`.

- `arma::Mat<dtype> ***bsdf** = nullptr` (optional output)<br>
  Principled BSDF (Bidirectional Scattering Distribution Function) values extracted from the
  .MTL file. Size `[mtl_names.size(), 17]`. Values are:
  0  | Base Color Red       | Range 0-1     | Default = 0.8
  1  | Base Color Green     | Range 0-1     | Default = 0.8
  2  | Base Color Blue      | Range 0-1     | Default = 0.8
  3  | Transparency (alpha) | Range 0-1     | Default = 1.0 (fully opaque)
  4  | Roughness            | Range 0-1     | Default = 0.5
  5  | Metallic             | Range 0-1     | Default = 0.0
  6  | Index of refraction (IOR)  | Range 0-4     | Default = 1.45
  7  | Specular Adjustment to the IOR | Range 0-1 | Default = 0.5 (no adjustment)
  8  | Emission Color Red    | Range 0-1     | Default = 0.0
  9  | Emission Color Green  | Range 0-1     | Default = 0.0
  10 | Emission Color Blue   | Range 0-1     | Default = 0.0
  11 | Sheen                 | Range 0-1     | Default = 0.0
  12 | Clearcoat             | Range 0-1     | Default = 0.0
  13 | Clearcoat roughness   | Range 0-1     | Default = 0.0
  14 | Anisotropic           | Range 0-1     | Default = 0.0
  15 | Anisotropic rotation  | Range 0-1     | Default = 0.0
  16 | Transmission          | Range 0-1     | Default = 0.0

## Returns:
- `arma::uword`<br>
  Number of mesh triangles found in the file (`n_mesh`).

## Technical Notes:
- Unknown or missing materials default to `"vacuum"` (ε_r = 1, σ = 0).
- Materials are applied per triangle via the `usemtl` tag in the `.obj` file.
- Input geometry must be fully triangulated—quads and n-gons are not supported.
- File parsing is case-sensitive for material names.

## Material Tag Format:
- Default materials (ITU-R P.2040-3 Table 3): `"usemtl itu_concrete"`, `"itu_brick"`, `"itu_wood"`, `"itu_water"`, etc.
- Frequency range: 1–40 GHz (limited to 1–10 GHz for ground materials)
- Custom materials syntax: `"usemtl Name::A:B:C:D:att"` with `A, B`: Real permittivity ε_r = `A * fGHz^B`,
  `C, D`: Conductivity σ = `C * fGHz^D`, `att`: Penetration loss in dB (fixed, per interaction)

## Material properties:
Each material is defined by its electrical properties. Radio waves that interact with a building will
produce losses that depend on the electrical properties of the building materials, the material
structure and the frequency of the radio wave. The fundamental quantities of interest are the electrical
permittivity (ϵ) and the conductivity (σ). A simple regression model for the frequency dependence is
obtained by fitting measured values of the permittivity and the conductivity at a number of frequencies.
The five parameters returned in `mtl_prop` then are:<br><br>

- Real part of relative permittivity at f = 1 GHz (a)
- Frequency dependence of rel. permittivity (b) such that ϵ = a · f^b
- Conductivity at f = 1 GHz (c)
- Frequency dependence of conductivity (d) such that σ = c· f^d
- Fixed attenuation in dB applied to each transition

A more detailed explanation together with a derivation can be found in ITU-R P.2040. The following
list of material is currently supported and the material can be selected by using the `usemtl` tag
in the OBJ file. When using Blender, the simply assign a material with that name to an object or face.
The following materials are defined by default:<br><br>

Name                  |         a |        b  |         c |         d |       Att |  max fGHz |
----------------------|-----------|-----------|-----------|-----------|-----------|-----------|
vacuum / air          |       1.0 |       0.0 |       0.0 |       0.0 |       0.0 |       100 |
textiles              |       1.5 |       0.0 |      5e-5 |      0.62 |       0.0 |       100 |
plastic               |      2.44 |       0.0 |   2.33e-5 |       1.0 |       0.0 |       100 |
ceramic               |       6.5 |       0.0 |    0.0023 |      1.32 |       0.0 |       100 |
sea_water             |      80.0 |     -0.25 |       4.0 |      0.58 |       0.0 |       100 |
sea_ice               |       3.2 |    -0.022 |       1.1 |       1.5 |       0.0 |       100 |
water                 |      80.0 |     -0.18 |       0.6 |      1.52 |       0.0 |        20 |
water_ice             |      3.17 |    -0.005 |    5.6e-5 |       1.7 |       0.0 |        20 |
itu_concrete          |      5.24 |       0.0 |    0.0462 |    0.7822 |       0.0 |       100 |
itu_brick             |      3.91 |       0.0 |    0.0238 |      0.16 |       0.0 |        40 |
itu_plasterboard      |      2.73 |       0.0 |    0.0085 |    0.9395 |       0.0 |       100 |
itu_wood              |      1.99 |       0.0 |    0.0047 |    1.0718 |       0.0 |       100 |
itu_glass             |      6.31 |       0.0 |    0.0036 |    1.3394 |       0.0 |       100 |
itu_ceiling_board     |      1.48 |       0.0 |    0.0011 |     1.075 |       0.0 |       100 |
itu_chipboard         |      2.58 |       0.0 |    0.0217 |      0.78 |       0.0 |       100 |
itu_plywood           |      2.71 |       0.0 |      0.33 |       0.0 |       0.0 |        40 |
itu_marble            |     7.074 |       0.0 |    0.0055 |    0.9262 |       0.0 |        60 |
itu_floorboard        |      3.66 |       0.0 |    0.0044 |    1.3515 |       0.0 |       100 |
itu_metal             |       1.0 |       0.0 |     1.0e7 |       0.0 |       0.0 |       100 |
itu_very_dry_ground   |       3.0 |       0.0 |   0.00015 |      2.52 |       0.0 |        10 |
itu_medium_dry_ground |      15.0 |      -0.1 |     0.035 |      1.63 |       0.0 |        10 |
itu_wet_ground        |      30.0 |      -0.4 |      0.15 |       1.3 |       0.0 |        10 |
itu_vegetation        |       1.0 |       0.0 |    1.0e-4 |       1.1 |       0.0 |       100 |
irr_glass             |      6.27 |       0.0 |    0.0043 |    1.1925 |      23.0 |       100 |


## Example:
```
arma::mat mesh, mtl_prop, vert_list;
arma::umat face_ind;
arma::uvec obj_ind, mtl_ind;
std::vector<std::string> obj_names, mtl_names;

quadriga_lib::obj_file_read<double>("cube.obj", &mesh, &mtl_prop, &vert_list, &face_ind, &obj_ind, &mtl_ind, &obj_names, &mtl_names);
```
MD!*/

// Read Wavefront .obj file
template <typename dtype>
arma::uword quadriga_lib::obj_file_read(std::string fn, arma::Mat<dtype> *mesh, arma::Mat<dtype> *mtl_prop, arma::Mat<dtype> *vert_list,
                                        arma::umat *face_ind, arma::uvec *obj_ind, arma::uvec *mtl_ind,
                                        std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names,
                                        arma::Mat<dtype> *bsdf)
{
    // Turn std::string into a std::filesystem::path
    std::filesystem::path obj_file{fn};

    if (!std::filesystem::exists(obj_file))
        throw std::invalid_argument("Error opening file: '" + fn + "' does not exist.");

    if (!std::filesystem::is_regular_file(obj_file))
        throw std::invalid_argument("Error opening file: '" + fn + "' is not a regular file.");

    // Open file for reading
    std::ifstream fileR{obj_file, std::ios::in};
    if (!fileR.is_open())
        throw std::invalid_argument("Error opening file: failed to open '" + fn + "'.");

    // Obtain the number of faces and vertices from the file
    arma::uword n_vert = 0, n_faces = 0;
    std::string line;
    while (std::getline(fileR, line))
        if (line.length() > 2 && line.at(0) == 118 && line.at(1) == 32) // Line starts with "v "
            ++n_vert;
        else if (line.length() > 2 && line.at(0) == 102) // Line starts with "f "
            ++n_faces;

    // Stop here if no other outputs are needed
    if (n_vert == 0 || n_faces == 0)
    {
        fileR.close();
        return 0;
    }

    if (mesh == nullptr && mtl_prop == nullptr && vert_list == nullptr && face_ind == nullptr && obj_ind == nullptr && mtl_ind == nullptr)
    {
        fileR.close();
        return n_faces;
    }

    // We need to clear exisiting object and material names, otherwise the indices will not match
    if (obj_names != nullptr)
        obj_names->clear();
    if (mtl_names != nullptr)
        mtl_names->clear();

    // Define a struct to store the material properties
    struct MaterialProp
    {
        std::string name;  // Material name
        double a, b, c, d; // Electromagnetic properties
        double att;        // Additional fixed  attenuation in dB
        arma::uword index; // Material index
    };

    // Add default material data, See: Rec. ITU-R P.2040-1, Table 3
    std::vector<MaterialProp> mtl_lib;
    mtl_lib.push_back({"vacuum", 1.0, 0.0, 0.0, 0.0, 0.0, 0});
    mtl_lib.push_back({"air", 1.0, 0.0, 0.0, 0.0, 0.0, 0});
    mtl_lib.push_back({"textiles", 1.5, 0.0, 5e-5, 0.62, 0.0, 0});
    mtl_lib.push_back({"plastic", 2.44, 0.0, 2.33e-5, 1.0, 0.0, 0});
    mtl_lib.push_back({"ceramic", 6.5, 0.0, 0.0023, 1.32, 0.0, 0});
    mtl_lib.push_back({"sea_water", 80.0, -0.25, 4.0, 0.58, 0.0, 0});
    mtl_lib.push_back({"sea_ice", 3.2, -0.022, 1.1, 1.5, 0.0, 0});
    mtl_lib.push_back({"water", 80.0, -0.18, 0.6, 1.52, 0.0, 0});
    mtl_lib.push_back({"water_ice", 3.17, -0.005, 5.6e-5, 1.7, 0.0, 0});
    mtl_lib.push_back({"itu_concrete", 5.24, 0.0, 0.0462, 0.7822, 0.0, 0});
    mtl_lib.push_back({"itu_brick", 3.91, 0.0, 0.0238, 0.16, 0.0, 0});
    mtl_lib.push_back({"itu_plasterboard", 2.73, 0.0, 0.0085, 0.9395, 0.0, 0});
    mtl_lib.push_back({"itu_wood", 1.99, 0.0, 0.0047, 1.0718, 0.0, 0});
    mtl_lib.push_back({"itu_glass", 6.31, 0.0, 0.0036, 1.3394, 0.0, 0});
    mtl_lib.push_back({"itu_ceiling_board", 1.48, 0.0, 0.0011, 1.075, 0.0, 0});
    mtl_lib.push_back({"itu_chipboard", 2.58, 0.0, 0.0217, 0.78, 0.0, 0});
    mtl_lib.push_back({"itu_plywood", 2.71, 0.0, 0.33, 0.0, 0.0, 0});
    mtl_lib.push_back({"itu_marble", 7.074, 0.0, 0.0055, 0.9262, 0.0, 0});
    mtl_lib.push_back({"itu_floorboard", 3.66, 0.0, 0.0044, 1.3515, 0.0, 0});
    mtl_lib.push_back({"itu_metal", 1.0, 0.0, 1.0e7, 0.0, 0.0, 0});
    mtl_lib.push_back({"itu_very_dry_ground", 3.0, 0.0, 0.00015, 2.52, 0.0, 0});
    mtl_lib.push_back({"itu_medium_dry_ground", 15.0, -0.1, 0.035, 1.63, 0.0, 0});
    mtl_lib.push_back({"itu_wet_ground", 30.0, -0.4, 0.15, 1.3, 0.0, 0});
    mtl_lib.push_back({"itu_vegetation", 1.0, 0.0, 1.0e-4, 1.1, 0.0, 0}); // Rec. ITU-R P.833-9, Figure 2
    mtl_lib.push_back({"irr_glass", 6.27, 0.0, 0.0043, 1.1925, 23.0, 0}); // 3GPP TR 38.901 V17.0.0, Table 7.4.3-1: Material penetration losses

    // Reset the file pointer to the beginning of the file
    fileR.clear(); // Clear any flags
    fileR.seekg(0, std::ios::beg);

    // Local data
    arma::uword i_vert = 0, i_face = 0, j_face = 0, i_object = 0, i_mtl = 0; // Counters for vertices, faces, objects, materials
    arma::uword iM = 0;                                                      // Material index
    double aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0;               // Default material properties
    bool simple_face_format = true;                                          // Selector for face format

    // Obtain memory for the vertex list
    dtype *p_vert;
    if (vert_list == nullptr)
        p_vert = new dtype[n_vert * 3];
    else if (vert_list->n_rows != n_vert || vert_list->n_cols != 3)
    {
        vert_list->set_size(n_vert, 3);
        p_vert = vert_list->memptr();
    }
    else
        p_vert = vert_list->memptr();

    // Obtain memory for face indices
    arma::uword *p_face_ind;
    if (face_ind == nullptr)
        p_face_ind = new arma::uword[n_faces * 3];
    else if (face_ind->n_rows != n_faces || face_ind->n_cols != 3)
    {
        face_ind->set_size(n_faces, 3);
        p_face_ind = face_ind->memptr();
    }
    else
        p_face_ind = face_ind->memptr();

    // Set size of "mtl_prop"
    if (mtl_prop != nullptr && (mtl_prop->n_rows != n_faces || mtl_prop->n_cols != 5))
        mtl_prop->set_size(n_faces, 5);
    dtype *p_mtl_prop = (mtl_prop == nullptr) ? nullptr : mtl_prop->memptr();

    // Set size of "mtl_ind"
    if (mtl_ind != nullptr && mtl_ind->n_elem != n_faces)
        mtl_ind->set_size(n_faces);
    arma::uword *p_mtl_ind = (mtl_ind == nullptr) ? nullptr : mtl_ind->memptr();

    // Set size of "obj_ind"
    if (obj_ind != nullptr && obj_ind->n_elem != n_faces)
        obj_ind->set_size(n_faces);
    arma::uword *p_obj_ind = (obj_ind == nullptr) ? nullptr : obj_ind->memptr();

    // Process file
    std::string mtllib_fn;
    while (std::getline(fileR, line))
    {
        // Read mtllib
        if (line.rfind("mtllib ", 0) == 0) // starts with "mtllib "
            mtllib_fn = line.substr(7);

        // Read vertex
        if (line.length() > 2 && line.at(0) == 118 && line.at(1) == 32) // Line starts with "v "
        {
            if (i_vert >= n_vert)
                throw std::invalid_argument("Error reading vertex data.");

            double x, y, z;
            std::sscanf(line.c_str(), "v %lf %lf %lf", &x, &y, &z);
            p_vert[i_vert] = (dtype)x;
            p_vert[i_vert + n_vert] = (dtype)y;
            p_vert[i_vert++ + 2 * n_vert] = (dtype)z;
        }

        // Read face
        else if (line.length() > 2 && line.at(0) == 102) // Line starts with "f "
        {
            if (i_face >= n_faces)
                throw std::invalid_argument("Error reading face data.");

            // Read face indices from file (1-based)
            arma::uword a = 0, b = 0, c = 0, d = 0;
            if (simple_face_format)
            {
                sscanf(line.c_str(), "f %llu %llu %llu %llu", &a, &b, &c, &d);
                simple_face_format = b != 0;
            }
            if (!simple_face_format)
                sscanf(line.c_str(), "f %llu%*[/0-9] %llu%*[/0-9] %llu%*[/0-9] %llu", &a, &b, &c, &d);

            if (a == 0 || b == 0 || c == 0)
                throw std::invalid_argument("Error reading face data.");

            if (d != 0)
                throw std::invalid_argument("Mesh is not in triangularized form.");

            // Store current material properties
            if (p_mtl_prop != nullptr)
                p_mtl_prop[i_face] = (dtype)aM,
                p_mtl_prop[i_face + n_faces] = (dtype)bM,
                p_mtl_prop[i_face + 2 * n_faces] = (dtype)cM,
                p_mtl_prop[i_face + 3 * n_faces] = (dtype)dM,
                p_mtl_prop[i_face + 4 * n_faces] = (dtype)attM;

            if (p_mtl_ind != nullptr)
                p_mtl_ind[i_face] = iM;

            // Store face indices (0-based)
            p_face_ind[i_face] = a - 1;
            p_face_ind[i_face + n_faces] = b - 1;
            p_face_ind[i_face++ + 2 * n_faces] = c - 1;
        }

        // Read objects ids (= connected faces)
        // - Object name is written to the OBJ file before vertices, materials and faces
        else if (line.length() > 2 && line.at(0) == 111) // Line starts with "o "
        {
            if (p_obj_ind != nullptr)
                for (arma::uword i = j_face; i < i_face; ++i)
                    p_obj_ind[i] = i_object;

            // Add object name to list of object names
            if (obj_names != nullptr)
            {
                std::string obj_name = line.substr(2, 255); // Name in OBJ File
                obj_names->push_back(obj_name);
            }

            // Reset current material
            aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, iM = 0;
            j_face = i_face;
            ++i_object;
        }

        // Read and set material properties
        // - Material names are written before face indices
        else if (line.length() > 7 && line.substr(0, 6).compare("usemtl") == 0) // Line contains material definition
        {
            std::string mtl_name = line.substr(7, 255);                 // Name in OBJ File
            aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, iM = 0; // Reset current material
            int found = -1;

            // If "mtl_name" does not contain a "::", remove everything after the dot
            if (mtl_name.find("::") == std::string::npos)
            {
                size_t dotPos = mtl_name.find('.');
                if (dotPos != std::string::npos)
                    mtl_name = mtl_name.substr(0, dotPos); // Substring up to the dot
            }

            // Try to find the material name in the material library
            for (size_t n = 0; n < mtl_lib.size(); ++n)
                if (mtl_lib[n].name.compare(mtl_name) == 0)
                {
                    aM = mtl_lib[n].a;
                    bM = mtl_lib[n].b;
                    cM = mtl_lib[n].c;
                    dM = mtl_lib[n].d;
                    attM = mtl_lib[n].att;
                    iM = mtl_lib[n].index;
                    found = (int)n;
                }

            if (found == -1) // Add new material
            {
                sscanf(mtl_name.c_str(), "%*[^:]::%lf:%lf:%lf:%lf:%lf", &aM, &bM, &cM, &dM, &attM);
                if (aM == 0.0)
                    mtl_lib.push_back({mtl_name, 1.0, 0.0, 0.0, 0.0, 0.0, 0}); // vacuum / air
                else
                    mtl_lib.push_back({mtl_name, aM, bM, cM, dM, attM, 0});
                found = (int)mtl_lib.size() - 1;
            }

            if (iM == 0) // Increase material counter
            {
                iM = ++i_mtl;
                mtl_lib[found].index = i_mtl;

                if (mtl_names != nullptr)
                    mtl_names->push_back(mtl_name);
            }
        }
    }

    // Set the object ID of the last object
    i_object = (i_object == 0) ? 1 : i_object; // Single unnamed object
    if (p_obj_ind != nullptr)
        for (arma::uword i = j_face; i < i_face; ++i)
            p_obj_ind[i] = i_object;

    // Calculate the triangle mesh from vertices and faces
    if (mesh != nullptr)
    {
        if (mesh->n_rows != n_faces || mesh->n_cols != 9)
            mesh->set_size(n_faces, 9);
        dtype *p_mesh = mesh->memptr();

        for (arma::uword n = 0; n < n_faces; ++n)
        {
            arma::uword a = p_face_ind[n],
                        b = p_face_ind[n + n_faces],
                        c = p_face_ind[n + 2 * n_faces];

            if (a > n_vert || b > n_vert || c > n_vert)
                throw std::invalid_argument("Error assembling triangle mesh.");

            p_mesh[n] = p_vert[a];
            p_mesh[n + n_faces] = p_vert[a + n_vert];
            p_mesh[n + 2 * n_faces] = p_vert[a + 2 * n_vert];
            p_mesh[n + 3 * n_faces] = p_vert[b];
            p_mesh[n + 4 * n_faces] = p_vert[b + n_vert];
            p_mesh[n + 5 * n_faces] = p_vert[b + 2 * n_vert];
            p_mesh[n + 6 * n_faces] = p_vert[c];
            p_mesh[n + 7 * n_faces] = p_vert[c + n_vert];
            p_mesh[n + 8 * n_faces] = p_vert[c + 2 * n_vert];
        }
    }

    // Clean up and return
    mtl_lib.clear();

    if (vert_list == nullptr)
        delete[] p_vert;

    if (face_ind == nullptr)
        delete[] p_face_ind;

    fileR.close();

    // Read BSDF data from MTL file
    if (bsdf != nullptr)
    {
        if (mtl_names == nullptr)
            throw std::invalid_argument("Cannot return 'mtl_color' without the corresponding 'mtl_names'.");

        std::filesystem::path mtl_file = obj_file;
        if (mtllib_fn.empty())
            mtl_file.replace_extension(".mtl");
        else
            mtl_file.replace_filename(mtllib_fn);

        if (!std::filesystem::exists(mtl_file))
        {
            bsdf->reset();
        }
        else
        {

            std::ifstream fileR{mtl_file, std::ios::in};
            if (!fileR.is_open())
                throw std::invalid_argument("Error opening file: failed to open '" + mtl_file.filename().string() + "'.");

            size_t n_mtl = mtl_names->size();
            if (bsdf->n_rows != n_mtl || bsdf->n_cols != 17)
                bsdf->set_size(n_mtl, 17);

            size_t i_mtl = 0;
            for (const auto &mtl : *mtl_names)
            {
                // Rewind to start
                fileR.clear();
                fileR.seekg(0);

                // Default values
                double R = 0.8, G = 0.8, B = 0.8;    // Base color
                double Re = 0.0, Ge = 0.0, Be = 0.0; // Emission color
                double alpha = 1.0;                  // Transparency
                double ior = 1.45;                   // Index of refraction
                double roughness = 0.5;
                double metallic = 0.0;
                double specular = 0.5;
                double sheen = 0.0;
                double clearcoat = 0.0;
                double clearcoat_roughness = 0.0;
                double anisotropic = 0.0;
                double anisotropic_rotation = 0.0;
                double transmission = 0.0;

                std::string line;
                bool foundMaterial = false;

                // Find the "newmtl <mtl>" line
                while (std::getline(fileR, line))
                    if (line.rfind("newmtl ", 0) == 0) // starts with "newmtl "
                    {
                        // extract everything after "newmtl "
                        std::string name = line.substr(7);
                        if (name == mtl)
                        {
                            foundMaterial = true;
                            break;
                        }
                    }
                if (!foundMaterial)
                {
                    throw std::invalid_argument(
                        "Error: material '" + mtl + "' not found in '" + fn + "'.");
                }

                // From here, scan until the next "newmtl " or EOF to look for Kd and Ns
                while (std::getline(fileR, line))
                {
                    if (line.rfind("newmtl ", 0) == 0)
                        break;

                    if (line.rfind("Kd ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> R >> G >> B;
                    }

                    if (line.rfind("Ke ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> Re >> Ge >> Be;
                    }

                    if (line.rfind("Ka ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> metallic;
                    }

                    if (line.rfind("Pm ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> metallic;
                    }

                    if (line.rfind("Ks ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> specular;
                    }

                    if (line.rfind("d ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(2));
                        iss >> alpha;
                    }

                    if (line.rfind("Ni ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> ior;
                    }

                    if (line.rfind("Ns ", 0) == 0)
                    {
                        double tmp;
                        std::istringstream iss(line.substr(3));
                        iss >> tmp;
                        roughness = 1.0 - std::sqrt(tmp * 0.001);
                    }

                    if (line.rfind("Pr ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> roughness;
                    }

                    if (line.rfind("Ps ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> sheen;
                    }

                    if (line.rfind("Pc ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> clearcoat;
                    }

                    if (line.rfind("Pcr ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(4));
                        iss >> clearcoat_roughness;
                    }

                    if (line.rfind("aniso ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(6));
                        iss >> anisotropic;
                    }

                    if (line.rfind("anisor ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(7));
                        iss >> anisotropic_rotation;
                    }

                    if (line.rfind("Tf ", 0) == 0)
                    {
                        std::istringstream iss(line.substr(3));
                        iss >> transmission;
                    }
                }

                // Fix ranges
                R = (R < 0.0) ? 0.0 : (R > 1.0 ? 1.0 : R);
                G = (G < 0.0) ? 0.0 : (G > 1.0 ? 1.0 : G);
                B = (B < 0.0) ? 0.0 : (B > 1.0 ? 1.0 : B);
                alpha = (alpha < 0.0) ? 0.0 : (alpha > 1.0 ? 1.0 : alpha);
                specular = (specular < 0.0) ? 0.0 : (specular > 1.0 ? 1.0 : specular);
                roughness = (roughness < 0.0) ? 0.0 : (roughness > 1.0 ? 1.0 : roughness);
                metallic = (metallic < 0.0) ? 0.0 : (metallic > 1.0 ? 1.0 : metallic);
                Re = (Re < 0.0) ? 0.0 : (Re > 1.0 ? 1.0 : Re);
                Ge = (Ge < 0.0) ? 0.0 : (Ge > 1.0 ? 1.0 : Ge);
                Be = (Be < 0.0) ? 0.0 : (Be > 1.0 ? 1.0 : Be);
                sheen = (sheen < 0.0) ? 0.0 : (sheen > 1.0 ? 1.0 : sheen);
                clearcoat = (clearcoat < 0.0) ? 0.0 : (clearcoat > 1.0 ? 1.0 : clearcoat);
                clearcoat_roughness = (clearcoat_roughness < 0.0) ? 0.0 : (clearcoat_roughness > 1.0 ? 1.0 : clearcoat_roughness);
                anisotropic = (anisotropic < 0.0) ? 0.0 : (anisotropic > 1.0 ? 1.0 : anisotropic);
                anisotropic_rotation = (anisotropic_rotation < 0.0) ? 0.0 : (anisotropic_rotation > 1.0 ? 1.0 : anisotropic_rotation);
                transmission = (transmission < 0.0) ? 0.0 : (transmission > 1.0 ? 1.0 : transmission);

                // Write to output
                bsdf->at(i_mtl, 0) = (dtype)R;
                bsdf->at(i_mtl, 1) = (dtype)G;
                bsdf->at(i_mtl, 2) = (dtype)B;
                bsdf->at(i_mtl, 3) = (dtype)alpha;
                bsdf->at(i_mtl, 4) = (dtype)roughness;
                bsdf->at(i_mtl, 5) = (dtype)metallic;
                bsdf->at(i_mtl, 6) = (dtype)ior;
                bsdf->at(i_mtl, 7) = (dtype)specular;
                bsdf->at(i_mtl, 8) = (dtype)Re;
                bsdf->at(i_mtl, 9) = (dtype)Ge;
                bsdf->at(i_mtl, 10) = (dtype)Be;
                bsdf->at(i_mtl, 11) = (dtype)sheen;
                bsdf->at(i_mtl, 12) = (dtype)clearcoat;
                bsdf->at(i_mtl, 13) = (dtype)clearcoat_roughness;
                bsdf->at(i_mtl, 14) = (dtype)anisotropic;
                bsdf->at(i_mtl, 15) = (dtype)anisotropic_rotation;
                bsdf->at(i_mtl, 16) = (dtype)transmission;
                ++i_mtl;
            }
        }

        fileR.close();
    }

    return n_faces;
}

template arma::uword quadriga_lib::obj_file_read(std::string fn, arma::Mat<float> *mesh, arma::Mat<float> *mtl_prop, arma::Mat<float> *vert_list,
                                                 arma::umat *face_ind, arma::uvec *obj_ind, arma::uvec *mtl_ind,
                                                 std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names,
                                                 arma::Mat<float> *bsdf);

template arma::uword quadriga_lib::obj_file_read(std::string fn, arma::Mat<double> *mesh, arma::Mat<double> *mtl_prop, arma::Mat<double> *vert_list,
                                                 arma::umat *face_ind, arma::uvec *obj_ind, arma::uvec *mtl_ind,
                                                 std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names,
                                                 arma::Mat<double> *bsdf);
