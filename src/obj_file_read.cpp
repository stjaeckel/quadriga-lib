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

// Read Wavefront .obj file
template <typename dtype>
arma::uword quadriga_lib::obj_file_read(std::string fn, arma::Mat<dtype> *mesh, arma::Mat<dtype> *mtl_prop, arma::Mat<dtype> *vert_list,
                                        arma::u32_mat *face_ind, arma::u32_vec *obj_ind, arma::u32_vec *mtl_ind,
                                        std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names)
{
    // Open file for reading
    std::ifstream fileR = std::ifstream(fn, std::ios::in);
    if (!fileR.is_open())
        throw std::invalid_argument("Error opening file.");

    // Obtain the number of faces and vertices from the file
    arma::uword n_vert = 0ULL, n_faces = 0ULL;
    std::string line;
    while (std::getline(fileR, line))
        if (line.length() > 2ULL && line.at(0ULL) == 118 && line.at(1ULL) == 32) // Line starts with "v "
            ++n_vert;
        else if (line.length() > 2ULL && line.at(0ULL) == 102) // Line starts with "f "
            ++n_faces;

    // Stop here if no other outputs are needed
    if (n_vert == 0ULL || n_faces == 0ULL)
    {
        fileR.close();
        return 0ULL;
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
    fileR.seekg(0ULL, std::ios::beg);

    // Local data
    arma::uword i_vert = 0ULL, i_face = 0ULL, j_face = 0ULL, i_object = 0ULL, i_mtl = 0ULL; // Counters for vertices, faces, objects, materials
    arma::uword iM = 0ULL;                                                                  // Material index
    double aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0;                              // Default material properties
    bool simple_face_format = true;                                                         // Selector for face format

    // Obtain memory for the vertex list
    dtype *p_vert;
    if (vert_list == nullptr)
        p_vert = new dtype[n_vert * 3ULL];
    else if (vert_list->n_rows != n_vert || vert_list->n_cols != 3ULL)
    {
        vert_list->set_size(n_vert, 3ULL);
        p_vert = vert_list->memptr();
    }
    else
        p_vert = vert_list->memptr();

    // Obtain memory for face indices
    unsigned *p_face_ind;
    if (face_ind == nullptr)
        p_face_ind = new unsigned[n_faces * 3ULL];
    else if (face_ind->n_rows != n_faces || face_ind->n_cols != 3ULL)
    {
        face_ind->set_size(n_faces, 3ULL);
        p_face_ind = face_ind->memptr();
    }
    else
        p_face_ind = face_ind->memptr();

    // Set size of "mtl_prop"
    if (mtl_prop != nullptr && (mtl_prop->n_rows != n_faces || mtl_prop->n_cols != 5ULL))
        mtl_prop->set_size(n_faces, 5ULL);
    dtype *p_mtl_prop = (mtl_prop == nullptr) ? nullptr : mtl_prop->memptr();

    // Set size of "mtl_ind"
    if (mtl_ind != nullptr && mtl_ind->n_elem != n_faces)
        mtl_ind->set_size(n_faces);
    unsigned *p_mtl_ind = (mtl_ind == nullptr) ? nullptr : mtl_ind->memptr();

    // Set size of "obj_ind"
    if (obj_ind != nullptr && obj_ind->n_elem != n_faces)
        obj_ind->set_size(n_faces);
    unsigned *p_obj_ind = (obj_ind == nullptr) ? nullptr : obj_ind->memptr();

    // Process file
    while (std::getline(fileR, line))
    {
        // Read vertex
        if (line.length() > 2ULL && line.at(0ULL) == 118 && line.at(1ULL) == 32) // Line starts with "v "
        {
            if (i_vert >= n_vert)
                throw std::invalid_argument("Error reading vertex data.");

            double x, y, z;
            std::sscanf(line.c_str(), "v %lf %lf %lf", &x, &y, &z);
            p_vert[i_vert] = (dtype)x;
            p_vert[i_vert + n_vert] = (dtype)y;
            p_vert[i_vert++ + 2ULL * n_vert] = (dtype)z;
        }

        // Read face
        else if (line.length() > 2ULL && line.at(0ULL) == 102) // Line starts with "f "
        {
            if (i_face >= n_faces)
                throw std::invalid_argument("Error reading face data.");

            // Read face indices from file (1-based)
            int a = 0, b = 0, c = 0, d = 0;
            if (simple_face_format)
            {
                sscanf(line.c_str(), "f %d %d %d %d", &a, &b, &c, &d);
                simple_face_format = b != 0;
            }
            if (!simple_face_format)
                sscanf(line.c_str(), "f %d%*[/0-9] %d%*[/0-9] %d%*[/0-9] %d", &a, &b, &c, &d);

            if (a == 0 || b == 0 || c == 0)
                throw std::invalid_argument("Error reading face data.");

            if (d != 0)
                throw std::invalid_argument("Mesh is not in triangularized form.");

            // Store current material properties
            if (p_mtl_prop != nullptr)
                p_mtl_prop[i_face] = (dtype)aM,
                p_mtl_prop[i_face + n_faces] = (dtype)bM,
                p_mtl_prop[i_face + 2ULL * n_faces] = (dtype)cM,
                p_mtl_prop[i_face + 3ULL * n_faces] = (dtype)dM,
                p_mtl_prop[i_face + 4ULL * n_faces] = (dtype)attM;

            if (p_mtl_ind != nullptr)
                p_mtl_ind[i_face] = (unsigned)iM;

            // Store face indices (0-based)
            p_face_ind[i_face] = (unsigned)a - 1;
            p_face_ind[i_face + n_faces] = (unsigned)b - 1;
            p_face_ind[i_face++ + 2ULL * n_faces] = (unsigned)c - 1;
        }

        // Read objects ids (= connected faces)
        // - Object name is written to the OBJ file before vertices, materials and faces
        else if (line.length() > 2ULL && line.at(0ULL) == 111) // Line starts with "o "
        {
            if (p_obj_ind != nullptr)
                for (arma::uword i = j_face; i < i_face; ++i)
                    p_obj_ind[i] = (unsigned)i_object;

            // Add object name to list of object names
            if (obj_names != nullptr)
            {
                std::string obj_name = line.substr(2ULL, 255ULL); // Name in OBJ File
                obj_names->push_back(obj_name);
            }

            // Reset current material
            aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, iM = 0ULL;
            j_face = i_face;
            ++i_object;
        }

        // Read and set material properties
        // - Material names are written before face indices
        else if (line.length() > 7ULL && line.substr(0ULL, 6ULL).compare("usemtl") == 0) // Line contains material definition
        {
            std::string mtl_name = line.substr(7ULL, 255ULL);              // Name in OBJ File
            aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, iM = 0ULL; // Reset current material
            int found = -1;

            // If "mtl_name" does not contain a "::", remove everything after the dot
            if (mtl_name.find("::") == std::string::npos)
            {
                size_t dotPos = mtl_name.find('.');
                if (dotPos != std::string::npos)
                    mtl_name = mtl_name.substr(0, dotPos); // Substring up to the dot
            }

            // Try to find the material name in the material library
            for (size_t n = 0ULL; n < mtl_lib.size(); ++n)
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

            if (iM == 0ULL) // Increase material counter
            {
                iM = ++i_mtl;
                mtl_lib[found].index = i_mtl;

                if (mtl_names != nullptr)
                    mtl_names->push_back(mtl_name);
            }
        }
    }

    // Set the object ID of the last object
    i_object = (i_object == 0ULL) ? 1ULL : i_object; // Single unnamed object
    if (p_obj_ind != nullptr)
        for (arma::uword i = j_face; i < i_face; ++i)
            p_obj_ind[i] = (unsigned)i_object;

    // Calculate the triangle mesh from vertices and faces
    if (mesh != nullptr)
    {
        if (mesh->n_rows != n_faces || mesh->n_cols != 9ULL)
            mesh->set_size(n_faces, 9ULL);
        dtype *p_mesh = mesh->memptr();

        for (arma::uword n = 0ULL; n < n_faces; ++n)
        {
            arma::uword a = p_face_ind[n],
                        b = p_face_ind[n + n_faces],
                        c = p_face_ind[n + 2ULL * n_faces];

            if (a > n_vert || b > n_vert || c > n_vert)
                throw std::invalid_argument("Error assembling triangle mesh.");

            p_mesh[n] = p_vert[a];
            p_mesh[n + n_faces] = p_vert[a + n_vert];
            p_mesh[n + 2ULL * n_faces] = p_vert[a + 2ULL * n_vert];
            p_mesh[n + 3ULL * n_faces] = p_vert[b];
            p_mesh[n + 4ULL * n_faces] = p_vert[b + n_vert];
            p_mesh[n + 5ULL * n_faces] = p_vert[b + 2ULL * n_vert];
            p_mesh[n + 6ULL * n_faces] = p_vert[c];
            p_mesh[n + 7ULL * n_faces] = p_vert[c + n_vert];
            p_mesh[n + 8ULL * n_faces] = p_vert[c + 2ULL * n_vert];
        }
    }

    // Clean up and return
    mtl_lib.clear();

    if (vert_list == nullptr)
        delete[] p_vert;

    if (face_ind == nullptr)
        delete[] p_face_ind;

    fileR.close();

    return n_faces;
}

template arma::uword quadriga_lib::obj_file_read(std::string fn, arma::Mat<float> *mesh, arma::Mat<float> *mtl_prop, arma::Mat<float> *vert_list,
                                                 arma::u32_mat *face_ind, arma::u32_vec *obj_ind, arma::u32_vec *mtl_ind,
                                                 std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names);

template arma::uword quadriga_lib::obj_file_read(std::string fn, arma::Mat<double> *mesh, arma::Mat<double> *mtl_prop, arma::Mat<double> *vert_list,
                                                 arma::u32_mat *face_ind, arma::u32_vec *obj_ind, arma::u32_vec *mtl_ind,
                                                 std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names);
