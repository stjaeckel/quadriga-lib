// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

// Struct to store the material properties
struct MaterialProp
{
    std::string name;
    double a, b, c, d;    // ε_r and σ models
    double att, attB;     // Att(f)   = att   * (f/fRef)^attB   [dB]
    double alpha, alphaB; // α(f)     = alpha * (f/fRef)^alphaB [dB/m]
    double fRef;          // reference frequency [GHz], default 1.0
    arma::uword index;
};

// Minimal CSV parser for material properties
// Required columns: name, a
// Optional columns (any order, any subset): b, c, d, att, attB, alpha, alphaB, fRef
// Missing/empty optional values default to 0.0 (1.0 for fRef)
static inline std::vector<MaterialProp> parse_materials_csv(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::invalid_argument("Error opening CSV file: '" + filename + "' does not exist.");

    std::vector<MaterialProp> materials;
    std::string line;

    // Read and parse header line
    if (!std::getline(file, line))
        throw std::invalid_argument("Error reading CSV file: '" + filename + "' is empty.");

    // Remove BOM if present (UTF-8)
    if (line.size() >= 3 && (unsigned char)line[0] == 0xEF &&
        (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
        line = line.substr(3);

    // Parse header to find column indices
    std::unordered_map<std::string, size_t> col_idx;
    {
        std::istringstream ss(line);
        std::string col;
        size_t idx = 0;
        while (std::getline(ss, col, ','))
        {
            size_t start = col.find_first_not_of(" \t\r\n");
            size_t end = col.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos)
                col = col.substr(start, end - start + 1);
            else
                col.clear();

            if (!col.empty())
                col_idx[col] = idx;
            idx++;
        }
    }

    // Validate required columns exist
    for (const auto &req : {"name", "a"})
        if (col_idx.find(req) == col_idx.end())
            throw std::invalid_argument("Error reading CSV file: '" + filename + "' is missing required column '" + std::string(req) + "'.");

    // Optional-column bookkeeping
    struct Col
    {
        bool present;
        size_t idx;
        double def;
    };
    auto make_opt = [&](const char *name, double def) -> Col
    {
        auto it = col_idx.find(name);
        return {it != col_idx.end(), it != col_idx.end() ? it->second : 0, def};
    };

    size_t idx_name = col_idx["name"];
    size_t idx_a = col_idx["a"];

    Col col_b = make_opt("b", 0.0);
    Col col_c = make_opt("c", 0.0);
    Col col_d = make_opt("d", 0.0);
    Col col_att = make_opt("att", 0.0);
    Col col_attB = make_opt("attB", 0.0);
    Col col_alpha = make_opt("alpha", 0.0);
    Col col_alphaB = make_opt("alphaB", 0.0);
    Col col_fRef = make_opt("fRef", 1.0);

    // Minimum required row width (covers all columns that *are* declared in the header)
    size_t max_idx = std::max(idx_name, idx_a);
    for (const Col *c : {&col_b, &col_c, &col_d, &col_att, &col_attB, &col_alpha, &col_alphaB, &col_fRef})
        if (c->present)
            max_idx = std::max(max_idx, c->idx);

    // Check for duplicate names while parsing
    std::unordered_set<std::string> seen_names;

    // Parse data rows
    size_t line_num = 1;
    while (std::getline(file, line))
    {
        line_num++;

        // Skip empty lines
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        // Parse row values
        std::vector<std::string> values;
        std::istringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ','))
        {
            size_t start = val.find_first_not_of(" \t\r\n");
            size_t end = val.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos)
                val = val.substr(start, end - start + 1);
            else
                val.clear();
            values.push_back(val);
        }

        // Require name and 'a' to be present in the row
        size_t required_max = std::max(idx_name, idx_a);
        if (values.size() <= required_max)
            throw std::invalid_argument("Error reading CSV file: '" + filename + "' line " +
                                        std::to_string(line_num) + " has insufficient columns.");

        // std::getline drops truly-trailing empty fields — pad so optional-column
        // access is in-bounds; padded empties fall through to defaults in parse_opt
        if (values.size() < max_idx + 1)
            values.resize(max_idx + 1);

        if (values.size() <= max_idx)
            throw std::invalid_argument("Error reading CSV file: '" + filename + "' line " + std::to_string(line_num) + " has insufficient columns.");

        // Parse optional double: default if column absent OR value cell empty
        auto parse_opt = [&](const Col &c) -> double
        {
            if (!c.present)
                return c.def;
            const std::string &s = values[c.idx];
            if (s.empty())
                return c.def;
            return std::stod(s);
        };

        MaterialProp mat;
        mat.name = values[idx_name];

        try
        {
            mat.a = std::stod(values[idx_a]); // required
            mat.b = parse_opt(col_b);
            mat.c = parse_opt(col_c);
            mat.d = parse_opt(col_d);
            mat.att = parse_opt(col_att);
            mat.attB = parse_opt(col_attB);
            mat.alpha = parse_opt(col_alpha);
            mat.alphaB = parse_opt(col_alphaB);
            mat.fRef = parse_opt(col_fRef);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument("Error reading CSV file: '" + filename + "' line " + std::to_string(line_num) + " contains invalid numeric value.");
        }

        if (mat.a <= 0.0)
            throw std::invalid_argument("Error reading CSV file: '" + filename + "' line " + std::to_string(line_num) + " has non-positive permittivity 'a'.");

        mat.index = 0;

        // Check for duplicate names
        if (!seen_names.insert(mat.name).second)
            throw std::invalid_argument("Error reading CSV file: Duplicate material name '" + mat.name + "' found.");

        materials.push_back(std::move(mat));
    }

    return materials;
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# obj_file_read
Read a Wavefront .obj file and extract geometry and material information

- Parses a triangulated Wavefront `.obj` file; quads and n-gons are not supported
- Materials applied per triangle via `usemtl` tag; unknown/missing materials default to `"vacuum"` (ε_r = 1, σ = 0, Att = 0, α = 0)
- Material name matching is case-sensitive
- Default materials follow ITU-R P.2040-3 Table 3 (1–40 GHz; ground materials limited to 1–10 GHz)
- Default material tag syntax: `usemtl itu_concrete` (or `itu_brick`, `itu_wood`, etc.)
- Custom material tag syntax: `usemtl Name::a:b:c:d:att:attB:alpha:alphaB:fRef`<br>
  - ε_r(f)   = a · (f/fRef)^b          (relative permittivity)<br>
  - σ(f)     = c · (f/fRef)^d    [S/m] (conductivity)<br>
  - Att(f)   = att · (f/fRef)^attB     [dB] (fixed penetration loss)<br>
  - α(f)     = alpha · (f/fRef)^alphaB [dB/m] (distance-dependent absorption)<br>
  - Trailing fields are optional; defaults are `b=c=d=att=attB=alpha=alphaB=0`, `fRef=1` GHz

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
    arma::Mat<dtype> *bsdf = nullptr,
    const std::string &materials_csv = "");
```

## Inputs:
- **`fn`** — Path to the `.obj` file
- **`materials_csv`** — Path to CSV file with custom material properties.
  Required columns: `name`, `a`. Optional columns: `b`, `c`, `d`, `att`, `attB`, `alpha`, `alphaB`, `fRef`.
  Column order is flexible; missing optional columns default to `0` (`fRef` → `1`).
  If empty, ITU-R P.2040-3 defaults are used.

## Outputs:
- **`mesh`** — Triangle vertex coordinates as `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per row; `[n_mesh, 9]`
- **`mtl_prop`** — Material properties; `[n_mesh, 9]`; Columns:<br><br>
  | Index | Symbol | Property                                      |
  | :---: | :----: | --------------------------------------------- |
  | 0     | a      | ε_r at fRef                                   |
  | 1     | b      | Frequency exponent for ε_r                    |
  | 2     | c      | σ at fRef [S/m]                               |
  | 3     | d      | Frequency exponent for σ                      |
  | 4     | att    | Penetration loss at fRef [dB]                 |
  | 5     | attB   | Frequency exponent for att                    |
  | 6     | alpha  | Distance absorption at fRef [dB/m]            |
  | 7     | alphaB | Frequency exponent for alpha                  |
  | 8     | fRef   | Reference frequency [GHz]                     |
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 0-based indices into `vert_list` per triangle; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; `[n_mesh]`
- **`mtl_ind`** — 1-based material index per triangle; `[n_mesh]`
- **`obj_names`** — Object names; length = `max(obj_ind)`
- **`mtl_names`** — Material names; length = `max(mtl_ind)`
- **`bsdf`** — Principled BSDF values from the `.mtl` file; `[n_mtl, 17]`; columns:<br><br>
   | Index | Property                  | Range | Default |
   | :---: | ------------------------- | :---: | ------: |
   | 0     | Base Color Red            | 0–1   | 0.8     |
   | 1     | Base Color Green          | 0–1   | 0.8     |
   | 2     | Base Color Blue           | 0–1   | 0.8     |
   | 3     | Transparency (alpha)      | 0–1   | 1.0     |
   | 4     | Roughness                 | 0–1   | 0.5     |
   | 5     | Metallic                  | 0–1   | 0.0     |
   | 6     | Index of refraction (IOR) | 0–4   | 1.45    |
   | 7     | Specular IOR adjustment   | 0–1   | 0.5     |
   | 8     | Emission Red              | 0–1   | 0.0     |
   | 9     | Emission Green            | 0–1   | 0.0     |
   | 10    | Emission Blue             | 0–1   | 0.0     |
   | 11    | Sheen                     | 0–1   | 0.0     |
   | 12    | Clearcoat                 | 0–1   | 0.0     |
   | 13    | Clearcoat roughness       | 0–1   | 0.0     |
   | 14    | Anisotropic               | 0–1   | 0.0     |
   | 15    | Anisotropic rotation      | 0–1   | 0.0     |
   | 16    | Transmission              | 0–1   | 0.0     |

## Returns:
- Number of triangular mesh elements (`n_mesh`)

## Default material table:
- For all defaults below: `attB = alpha = alphaB = 0` and `fRef = 1 GHz`:<br><br>
  | Name                  | a     | b      | c       | d      | att  | max fGHz |
  | --------------------- | ----: | -----: | ------: | -----: | ---: | -------: |
  | vacuum / air          | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 100      |
  | textiles              | 1.5   | 0.0    | 5e-5    | 0.62   | 0.0  | 100      |
  | plastic               | 2.44  | 0.0    | 2.33e-5 | 1.0    | 0.0  | 100      |
  | ceramic               | 6.5   | 0.0    | 0.0023  | 1.32   | 0.0  | 100      |
  | sea_water             | 80.0  | -0.25  | 4.0     | 0.58   | 0.0  | 100      |
  | sea_ice               | 3.2   | -0.022 | 1.1     | 1.5    | 0.0  | 100      |
  | water                 | 80.0  | -0.18  | 0.6     | 1.52   | 0.0  | 20       |
  | water_ice             | 3.17  | -0.005 | 5.6e-5  | 1.7    | 0.0  | 20       |
  | itu_concrete          | 5.24  | 0.0    | 0.0462  | 0.7822 | 0.0  | 100      |
  | itu_brick             | 3.91  | 0.0    | 0.0238  | 0.16   | 0.0  | 40       |
  | itu_plasterboard      | 2.73  | 0.0    | 0.0085  | 0.9395 | 0.0  | 100      |
  | itu_wood              | 1.99  | 0.0    | 0.0047  | 1.0718 | 0.0  | 100      |
  | itu_glass             | 6.31  | 0.0    | 0.0036  | 1.3394 | 0.0  | 100      |
  | itu_ceiling_board     | 1.48  | 0.0    | 0.0011  | 1.075  | 0.0  | 100      |
  | itu_chipboard         | 2.58  | 0.0    | 0.0217  | 0.78   | 0.0  | 100      |
  | itu_plywood           | 2.71  | 0.0    | 0.33    | 0.0    | 0.0  | 40       |
  | itu_marble            | 7.074 | 0.0    | 0.0055  | 0.9262 | 0.0  | 60       |
  | itu_floorboard        | 3.66  | 0.0    | 0.0044  | 1.3515 | 0.0  | 100      |
  | itu_metal             | 1.0   | 0.0    | 1.0e7   | 0.0    | 0.0  | 100      |
  | itu_very_dry_ground   | 3.0   | 0.0    | 0.00015 | 2.52   | 0.0  | 10       |
  | itu_medium_dry_ground | 15.0  | -0.1   | 0.035   | 1.63   | 0.0  | 10       |
  | itu_wet_ground        | 30.0  | -0.4   | 0.15    | 1.3    | 0.0  | 10       |
  | itu_vegetation        | 1.0   | 0.0    | 1.0e-4  | 1.1    | 0.0  | 100      |
  | irr_glass             | 6.27  | 0.0    | 0.0043  | 1.1925 | 23.0 | 100      |

## See also:
- [[obj_file_write]] (for writing OBJ files)
- [[obj_overlap_test]] (for testing mesh geometry)
- [[triangle_mesh_segmentation]] (used to calculate indexed mesh for faster processing)
- [[ray_mesh_interact]] (calculating interactions between rays and the triangular mesh)
- [[mitsuba_xml_file_write]] (for exporting to Mitsuba scene file format)
MD!*/

template <typename dtype>
arma::uword quadriga_lib::obj_file_read(std::string fn, arma::Mat<dtype> *mesh, arma::Mat<dtype> *mtl_prop, arma::Mat<dtype> *vert_list,
                                        arma::umat *face_ind, arma::uvec *obj_ind, arma::uvec *mtl_ind,
                                        std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names,
                                        arma::Mat<dtype> *bsdf, const std::string &materials_csv)
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

    // We need to clear existing object and material names, otherwise the indices will not match
    if (obj_names != nullptr)
        obj_names->clear();
    if (mtl_names != nullptr)
        mtl_names->clear();

    // Define materials
    std::vector<MaterialProp> mtl_lib;
    if (materials_csv.empty()) // Use default materials
    {
        // Add default material data, See: Rec. ITU-R P.2040-1, Table 3
        mtl_lib.push_back({"vacuum", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"air", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"textiles", 1.5, 0.0, 5e-5, 0.62, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"plastic", 2.44, 0.0, 2.33e-5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"ceramic", 6.5, 0.0, 0.0023, 1.32, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"sea_water", 80.0, -0.25, 4.0, 0.58, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"sea_ice", 3.2, -0.022, 1.1, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"water", 80.0, -0.18, 0.6, 1.52, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"water_ice", 3.17, -0.005, 5.6e-5, 1.7, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_concrete", 5.24, 0.0, 0.0462, 0.7822, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_brick", 3.91, 0.0, 0.0238, 0.16, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_plasterboard", 2.73, 0.0, 0.0085, 0.9395, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_wood", 1.99, 0.0, 0.0047, 1.0718, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_glass", 6.31, 0.0, 0.0036, 1.3394, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_ceiling_board", 1.48, 0.0, 0.0011, 1.075, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_chipboard", 2.58, 0.0, 0.0217, 0.78, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_plywood", 2.71, 0.0, 0.33, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_marble", 7.074, 0.0, 0.0055, 0.9262, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_floorboard", 3.66, 0.0, 0.0044, 1.3515, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_metal", 1.0, 0.0, 1.0e7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_very_dry_ground", 3.0, 0.0, 0.00015, 2.52, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_medium_dry_ground", 15.0, -0.1, 0.035, 1.63, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_wet_ground", 30.0, -0.4, 0.15, 1.3, 0.0, 0.0, 0.0, 0.0, 1.0, 0});
        mtl_lib.push_back({"itu_vegetation", 1.0, 0.0, 1.0e-4, 1.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0}); // Rec. ITU-R P.833-9, Figure 2
        mtl_lib.push_back({"irr_glass", 6.27, 0.0, 0.0043, 1.1925, 23.0, 0.0, 0.0, 0.0, 1.0, 0}); // 3GPP TR 38.901 V17.0.0, Table 7.4.3-1: Material penetration losses
    }
    else if (!std::filesystem::exists(materials_csv)) // Check if a given CSV file exists
        throw std::invalid_argument("Error opening file: CSV file '" + materials_csv + "' does not exist.");
    else // Import data from CSV file
        mtl_lib = parse_materials_csv(materials_csv);

    // Reset the file pointer to the beginning of the file
    fileR.clear(); // Clear any flags
    fileR.seekg(0, std::ios::beg);

    // Local data
    arma::uword i_vert = 0, i_face = 0, j_face = 0, i_object = 0, i_mtl = 0; // Counters for vertices, faces, objects, materials
    arma::uword iM = 0;                                                      // Material index
    double aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, fRefM = 1.0;              // Default material properties
    double attM = 0.0, attBM = 0.0, alphaM = 0.0, alphaBM = 0.0;             // Default attenuation properties
    bool simple_face_format = true;                                          // Selector for face format

    // Obtain memory for the vertex list (scratch buffer if caller doesn't request it)
    std::vector<dtype> vert_scratch;
    dtype *p_vert;
    if (vert_list == nullptr)
    {
        vert_scratch.resize(n_vert * 3);
        p_vert = vert_scratch.data();
    }
    else
    {
        if (vert_list->n_rows != n_vert || vert_list->n_cols != 3)
            vert_list->set_size(n_vert, 3);
        p_vert = vert_list->memptr();
    }

    // Obtain memory for face indices (scratch buffer if caller doesn't request it)
    std::vector<arma::uword> face_ind_scratch;
    arma::uword *p_face_ind;
    if (face_ind == nullptr)
    {
        face_ind_scratch.resize(n_faces * 3);
        p_face_ind = face_ind_scratch.data();
    }
    else
    {
        if (face_ind->n_rows != n_faces || face_ind->n_cols != 3)
            face_ind->set_size(n_faces, 3);
        p_face_ind = face_ind->memptr();
    }

    // Set size of "mtl_prop"
    if (mtl_prop != nullptr && (mtl_prop->n_rows != n_faces || mtl_prop->n_cols != 9))
        mtl_prop->set_size(n_faces, 9);
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
        {
            mtllib_fn = line.substr(7);
            mtllib_fn.erase(mtllib_fn.find_last_not_of(" \t\r\n") + 1); // Trim trailing whitespace / CR
        }

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
            {
                p_mtl_prop[i_face] = (dtype)aM;
                p_mtl_prop[i_face + n_faces] = (dtype)bM;
                p_mtl_prop[i_face + 2 * n_faces] = (dtype)cM;
                p_mtl_prop[i_face + 3 * n_faces] = (dtype)dM;
                p_mtl_prop[i_face + 4 * n_faces] = (dtype)attM;
                p_mtl_prop[i_face + 5 * n_faces] = (dtype)attBM;
                p_mtl_prop[i_face + 6 * n_faces] = (dtype)alphaM;
                p_mtl_prop[i_face + 7 * n_faces] = (dtype)alphaBM;
                p_mtl_prop[i_face + 8 * n_faces] = (dtype)fRefM;
            }

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
            aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, attBM = 0.0, alphaM = 0.0, alphaBM = 0.0, fRefM = 1.0, iM = 0;
            j_face = i_face;
            ++i_object;
        }

        // Read and set material properties
        // - Material names are written before face indices
        else if (line.length() > 7 && line.substr(0, 6).compare("usemtl") == 0) // Line contains material definition
        {
            std::string mtl_name = line.substr(7, 255); // Name in OBJ File
            std::string mtl_name_raw = mtl_name;

            // Reset current material
            aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, attBM = 0.0, alphaM = 0.0, alphaBM = 0.0, fRefM = 1.0, iM = 0;
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
                    attBM = mtl_lib[n].attB;
                    alphaM = mtl_lib[n].alpha;
                    alphaBM = mtl_lib[n].alphaB;
                    fRefM = mtl_lib[n].fRef;
                    iM = mtl_lib[n].index;
                    found = (int)n;
                }

            if (found == -1) // Add new material
            {
                aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, attBM = 0.0, alphaM = 0.0, alphaBM = 0.0, fRefM = 1.0;
                sscanf(mtl_name.c_str(),
                       "%*[^:]::%lf:%lf:%lf:%lf:%lf:%lf:%lf:%lf:%lf",
                       &aM, &bM, &cM, &dM, &attM, &attBM, &alphaM, &alphaBM, &fRefM);
                if (aM <= 0.0) // ε_r can't be ≤ 0; reset the permittivity pair
                    aM = 1.0, bM = 0.0;
                mtl_lib.push_back({mtl_name, aM, bM, cM, dM, attM, attBM, alphaM, alphaBM, fRefM, 0});
                found = (int)mtl_lib.size() - 1;
            }

            if (iM == 0) // Increase material counter
            {
                iM = ++i_mtl;
                mtl_lib[found].index = i_mtl;

                if (mtl_names != nullptr)
                    mtl_names->push_back(mtl_name_raw);
            }
        }
    }

    // Set the object ID of the last object
    i_object = (i_object == 0) ? 1 : i_object; // Single unnamed object
    if (p_obj_ind != nullptr)
        for (arma::uword i = j_face; i < i_face; ++i)
            p_obj_ind[i] = i_object;

    // Single unnamed object: ensure obj_names has one entry
    if (obj_names != nullptr && i_object == 1 && obj_names->empty())
        obj_names->push_back("object");

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
    fileR.close();

    // Read BSDF data from MTL file
    if (bsdf != nullptr)
    {
        if (mtl_names == nullptr)
            throw std::invalid_argument("Cannot return 'bsdf' without the corresponding 'mtl_names'.");

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
                    // Custom inline materials (:: syntax) won't appear in .mtl file; skip with defaults
                    ++i_mtl;
                    continue;
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
                                                 arma::Mat<float> *bsdf, const std::string &materials_csv);

template arma::uword quadriga_lib::obj_file_read(std::string fn, arma::Mat<double> *mesh, arma::Mat<double> *mtl_prop, arma::Mat<double> *vert_list,
                                                 arma::umat *face_ind, arma::uvec *obj_ind, arma::uvec *mtl_ind,
                                                 std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names,
                                                 arma::Mat<double> *bsdf, const std::string &materials_csv);
