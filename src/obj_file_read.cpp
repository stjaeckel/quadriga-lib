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
#include <filesystem>
#include <cstdio>
#include <cmath>

// Built-in EM/acoustic material table (Rec. ITU-R P.2040-3; irr_glass: 3GPP TR 38.901 V17.0.0, Table 7.4.3-1).
// Stored as a CSV literal so it is parsed through the exact same path as a user-supplied CSV file.
// Row 0 is the transparent fallback used when a referenced material is not found and csv_strict == false.
// Only columns carrying non-default data are listed; consumers supply their own defaults for absent columns.
static const char *DEFAULT_MATERIAL_CSV =
    "name,a,b,c,d,att\n"
    "air,1.0,0.0,0.0,0.0,0.0\n"
    "vacuum,1.0,0.0,0.0,0.0,0.0\n"
    "textiles,1.5,0.0,5e-5,0.62,0.0\n"
    "plastic,2.44,0.0,2.33e-5,1.0,0.0\n"
    "ceramic,6.5,0.0,0.0023,1.32,0.0\n"
    "sea_water,80.0,-0.25,4.0,0.58,0.0\n"
    "sea_ice,3.2,-0.022,1.1,1.5,0.0\n"
    "water,80.0,-0.18,0.6,1.52,0.0\n"
    "water_ice,3.17,-0.005,5.6e-5,1.7,0.0\n"
    "itu_concrete,5.24,0.0,0.0462,0.7822,0.0\n"
    "itu_brick,3.91,0.0,0.0238,0.16,0.0\n"
    "itu_plasterboard,2.73,0.0,0.0085,0.9395,0.0\n"
    "itu_wood,1.99,0.0,0.0047,1.0718,0.0\n"
    "itu_glass,6.31,0.0,0.0036,1.3394,0.0\n"
    "itu_ceiling_board,1.48,0.0,0.0011,1.075,0.0\n"
    "itu_chipboard,2.58,0.0,0.0217,0.78,0.0\n"
    "itu_plywood,2.71,0.0,0.33,0.0,0.0\n"
    "itu_marble,7.074,0.0,0.0055,0.9262,0.0\n"
    "itu_floorboard,3.66,0.0,0.0044,1.3515,0.0\n"
    "itu_metal,1.0,0.0,1.0e7,0.0,0.0\n"
    "itu_very_dry_ground,3.0,0.0,0.00015,2.52,0.0\n"
    "itu_medium_dry_ground,15.0,-0.1,0.035,1.63,0.0\n"
    "itu_wet_ground,30.0,-0.4,0.15,1.3,0.0\n"
    "itu_vegetation,1.0,0.0,1.0e-4,1.1,0.0\n"
    "irr_glass,6.27,0.0,0.0043,1.1925,23.0\n";

// Parsed material table. Numeric columns are kept in CSV order; values[col][row] are stored as double
// and cast to dtype on output. The name->row map drives material lookup from the .obj file.
struct MtlTable
{
    std::vector<std::string> names;                     // row -> material name; length n_csv
    std::vector<std::string> columns;                   // numeric column names (excludes "name"), CSV order
    std::vector<std::vector<double>> values;            // values[col][row]
    std::unordered_map<std::string, arma::uword> index; // name -> row
};

// Trim leading/trailing whitespace (incl. CR/LF/tab)
static inline std::string trim_ws(const std::string &s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
        return std::string();
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Split a CSV line into trimmed cells.
// Note: std::getline drops truly-trailing empty fields, so callers pad rows to the header width before indexing.
static inline std::vector<std::string> split_csv(const std::string &line)
{
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
        out.push_back(trim_ws(cell));
    return out;
}

// Return the base material name: everything before the first '.' is the CSV lookup key.
// "itu_concrete.001" -> "itu_concrete"; "concrete.gray.001" -> "concrete".
// Names without a dot are returned unchanged. This lets one CSV material back many
// Blender sub-materials (concrete.gray, concrete.white, ... and their .001 duplicates).
static inline std::string base_material_name(const std::string &s)
{
    size_t dot = s.find('.');
    return (dot == std::string::npos) ? s : s.substr(0, dot);
}

// Parse a material table from any input stream. The reader is schema-blind: it requires only a
// "name" column (the join key for .obj materials); every other column is treated as numeric.
// Empty numeric cells parse as 0.0. Non-numeric cells in a numeric column are an error.
static MtlTable parse_mtl_table(std::istream &in, const std::string &src)
{
    MtlTable T;
    std::string line;

    if (!std::getline(in, line))
        throw std::invalid_argument("Error reading materials: '" + src + "' is empty.");

    // Strip UTF-8 BOM if present
    if (line.size() >= 3 && (unsigned char)line[0] == 0xEF &&
        (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
        line.erase(0, 3);

    std::vector<std::string> header = split_csv(line);
    if (header.empty())
        throw std::invalid_argument("Error reading materials: '" + src + "' has an empty header.");

    // Locate the mandatory "name" column
    arma::uword name_col = (arma::uword)header.size();
    for (arma::uword k = 0; k < (arma::uword)header.size(); ++k)
        if (header[k] == "name")
        {
            name_col = k;
            break;
        }
    if (name_col == (arma::uword)header.size())
        throw std::invalid_argument("Error reading materials: '" + src + "' is missing required column 'name'.");

    // Collect numeric columns, preserving CSV order
    std::vector<arma::uword> num_pos;
    for (arma::uword k = 0; k < (arma::uword)header.size(); ++k)
        if (k != name_col)
        {
            if (header[k].empty())
                throw std::invalid_argument("Error reading materials: '" + src + "' has an empty column name in the header.");
            T.columns.push_back(header[k]);
            num_pos.push_back(k);
        }
    T.values.resize(T.columns.size());

    std::unordered_set<std::string> seen;
    arma::uword line_num = 1;

    while (std::getline(in, line))
    {
        ++line_num;
        if (trim_ws(line).empty())
            continue;

        std::vector<std::string> cell = split_csv(line);
        if (cell.size() < header.size())
            cell.resize(header.size()); // pad trailing empties dropped by getline

        const std::string &nm = cell[name_col];
        if (nm.empty())
            throw std::invalid_argument("Error reading materials: '" + src + "' line " +
                                        std::to_string(line_num) + " has an empty material name.");
        if (!seen.insert(nm).second)
            throw std::invalid_argument("Error reading materials: duplicate material name '" + nm + "' in '" + src + "'.");

        T.index[nm] = (arma::uword)T.names.size();
        T.names.push_back(nm);

        for (size_t c = 0; c < T.columns.size(); ++c)
        {
            const std::string &v = cell[num_pos[c]];
            double d = 0.0; // empty cell -> 0.0
            if (!v.empty())
            {
                try
                {
                    d = std::stod(v);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Error reading materials: '" + src + "' line " +
                                                std::to_string(line_num) + " column '" + T.columns[c] +
                                                "' is not numeric ('" + v + "').");
                }
            }
            T.values[c].push_back(d);
        }
    }

    if (T.names.empty())
        throw std::invalid_argument("Error reading materials: '" + src + "' contains no material rows.");

    return T;
}

// Load the material table: built-in default if fn_csv is empty, otherwise from the CSV file.
static MtlTable load_mtl_table(const std::string &fn_csv)
{
    if (fn_csv.empty())
    {
        std::istringstream ss(DEFAULT_MATERIAL_CSV);
        return parse_mtl_table(ss, "<default materials>");
    }
    if (!std::filesystem::exists(fn_csv))
        throw std::invalid_argument("Error opening file: CSV file '" + fn_csv + "' does not exist.");
    std::ifstream f(fn_csv);
    if (!f.is_open())
        throw std::invalid_argument("Error opening file: failed to open CSV file '" + fn_csv + "'.");
    return parse_mtl_table(f, fn_csv);
}

// Read Principled BSDF (visual) properties from the companion .mtl file, one row per name in 'names'
// (matched against "newmtl" by raw name). Columns: R,G,B,alpha,roughness,metallic,ior,specular,
// Re,Ge,Be,sheen,clearcoat,clearcoat_roughness,anisotropic,anisotropic_rotation,transmission.
// If the .mtl is absent or no name matches, bsdf is returned empty ([0,17]).
template <typename dtype>
static void read_mtl_bsdf(const std::filesystem::path &obj_file, const std::string &mtllib_fn,
                          const std::vector<std::string> &names, arma::Mat<dtype> *bsdf)
{
    std::filesystem::path mtl_file = obj_file;
    if (mtllib_fn.empty())
        mtl_file.replace_extension(".mtl");
    else
        mtl_file.replace_filename(mtllib_fn);

    if (names.empty() || !std::filesystem::exists(mtl_file))
    {
        bsdf->set_size(0, 17);
        return;
    }

    std::ifstream fileR{mtl_file, std::ios::in};
    if (!fileR.is_open())
        throw std::invalid_argument("Error opening file: failed to open '" + mtl_file.filename().string() + "'.");

    const size_t n_mtl = names.size();
    bsdf->set_size(n_mtl, 17);

    bool any_found = false;
    for (size_t i_mtl = 0; i_mtl < n_mtl; ++i_mtl)
    {
        fileR.clear();
        fileR.seekg(0, std::ios::beg);

        // Defaults
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

        // Find "newmtl <name>"
        while (std::getline(fileR, line))
            if (line.rfind("newmtl ", 0) == 0 && line.substr(7) == names[i_mtl])
            {
                any_found = true;
                break;
            }

        // Scan until the next "newmtl " or EOF
        while (std::getline(fileR, line))
        {
            if (line.rfind("newmtl ", 0) == 0)
                break;

            if (line.rfind("Kd ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> R >> G >> B;
            }
            else if (line.rfind("Ke ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> Re >> Ge >> Be;
            }
            else if (line.rfind("Ka ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> metallic;
            }
            else if (line.rfind("Pm ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> metallic;
            }
            else if (line.rfind("Ks ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> specular;
            }
            else if (line.rfind("d ", 0) == 0)
            {
                std::istringstream iss(line.substr(2));
                iss >> alpha;
            }
            else if (line.rfind("Ni ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> ior;
            }
            else if (line.rfind("Ns ", 0) == 0)
            {
                double tmp = 0.0;
                std::istringstream iss(line.substr(3));
                iss >> tmp;
                roughness = 1.0 - std::sqrt(tmp * 0.001);
            }
            else if (line.rfind("Pr ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> roughness;
            }
            else if (line.rfind("Ps ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> sheen;
            }
            else if (line.rfind("Pcr ", 0) == 0) // check before "Pc " (prefix overlap)
            {
                std::istringstream iss(line.substr(4));
                iss >> clearcoat_roughness;
            }
            else if (line.rfind("Pc ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> clearcoat;
            }
            else if (line.rfind("anisor ", 0) == 0) // check before "aniso " (prefix overlap)
            {
                std::istringstream iss(line.substr(7));
                iss >> anisotropic_rotation;
            }
            else if (line.rfind("aniso ", 0) == 0)
            {
                std::istringstream iss(line.substr(6));
                iss >> anisotropic;
            }
            else if (line.rfind("Tf ", 0) == 0)
            {
                std::istringstream iss(line.substr(3));
                iss >> transmission;
            }
        }

        // Clamp to [0,1] (ior left unconstrained)
        auto clamp01 = [](double x)
        { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };
        R = clamp01(R);
        G = clamp01(G);
        B = clamp01(B);
        alpha = clamp01(alpha);
        specular = clamp01(specular);
        roughness = clamp01(roughness);
        metallic = clamp01(metallic);
        Re = clamp01(Re);
        Ge = clamp01(Ge);
        Be = clamp01(Be);
        sheen = clamp01(sheen);
        clearcoat = clamp01(clearcoat);
        clearcoat_roughness = clamp01(clearcoat_roughness);
        anisotropic = clamp01(anisotropic);
        anisotropic_rotation = clamp01(anisotropic_rotation);
        transmission = clamp01(transmission);

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
    }

    if (!any_found)
        bsdf->set_size(0, 17);
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

- Parses a triangulated `.obj`; quads and n-gons are rejected. Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by name with a trailing Blender
  `.NNN` suffix removed. Unmatched names throw when `csv_strict = true`; otherwise they map to row 0
  of the table (the transparent fallback). The two index spaces are decoupled, so several visual
  materials (e.g. `wall.001`, `wall.002`) may resolve to a single EM material.
- All returned indices are 0-based.
- With an empty `fn_obj`, geometry and `.mtl` outputs are empty and only the table (`csv_names`, `csv_prop`) 
  is populated — useful for inspecting a CSV or the default library. If `fn_csv` is also empty, the built-in default table is returned.
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>
  
## Declaration:
```
arma::uword quadriga_lib::obj_file_read(
    const std::string &fn_obj = "",
    arma::Mat<dtype> *mesh = nullptr,
    arma::Mat<dtype> *vert_list = nullptr,
    arma::umat *face_ind = nullptr,
    arma::uvec *obj_ind = nullptr,
    std::vector<std::string> *obj_names = nullptr,
    arma::uvec *mtl_ind = nullptr,
    std::vector<std::string> *mtl_names = nullptr,
    arma::Mat<dtype> *bsdf = nullptr,
    const std::string &fn_csv = "",
    arma::uvec *csv_ind = nullptr,
    std::vector<std::string> *csv_names = nullptr,
    std::unordered_map<std::string, std::vector<dtype>> *csv_prop = nullptr,
    bool csv_strict = false);
```

## Inputs:
- **`fn_obj`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** — Path to an EM/acoustic material CSV; must contain a `name` column, and row 0 is the
  fallback material (should be transparent, e.g. air). Empty uses the built-in ITU-R P.2040 default table.
- **`csv_strict`** — If `true`, throw when a `usemtl` material is absent from the table; otherwise map to row 0

## Outputs:
- **`mesh`** — Triangle vertex coordinates `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per row; `[n_mesh, 9]`
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 0-based vertex indices into `vert_list` per triangle; `[n_mesh, 3]`
- **`obj_ind`** — 0-based object index per triangle; `[n_mesh]`
- **`obj_names`** — Object names; length `max(obj_ind)+1`
- **`mtl_ind`** — 0-based visual-material index per triangle; `[n_mesh]`
- **`mtl_names`** — Visual material names (raw `usemtl`); length `no_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `[no_mtl, 17]`
- **`csv_ind`** — 0-based EM/acoustic-material index per triangle; `[n_mesh]`
- **`csv_names`** — Material names from the table; length `n_csv` (the full table)
- **`csv_prop`** — Material properties keyed by CSV column name (excluding `name`); each value has
  length `n_csv`. Columns absent from the table are defaulted by consumers; empty cells parse as 0.

## Returns:
- Number of triangular mesh elements (`n_mesh`)

## See also:
- [[obj_file_write]] (for writing OBJ files)
- [[obj_overlap_test]] (for testing mesh geometry)
- [[triangle_mesh_segmentation]] (used to calculate indexed mesh for faster processing)
- [[ray_mesh_interact]] (calculating interactions between rays and the triangular mesh)
- [[mitsuba_xml_file_write]] (for exporting to Mitsuba scene file format)
MD!*/

template <typename dtype>
arma::uword quadriga_lib::obj_file_read(const std::string &fn_obj, arma::Mat<dtype> *mesh, arma::Mat<dtype> *vert_list,
                                        arma::umat *face_ind, arma::uvec *obj_ind, std::vector<std::string> *obj_names,
                                        arma::uvec *mtl_ind, std::vector<std::string> *mtl_names, arma::Mat<dtype> *bsdf,
                                        const std::string &fn_csv, arma::uvec *csv_ind, std::vector<std::string> *csv_names,
                                        std::unordered_map<std::string, std::vector<dtype>> *csv_prop, bool csv_strict)
{
    const bool want_csv_table = (csv_ind != nullptr || csv_names != nullptr || csv_prop != nullptr);

    // Material table (loaded on demand). Lives across the whole function for the lambdas below.
    MtlTable table;

    // Copy the table's column outputs to csv_names / csv_prop
    auto fill_csv_table_out = [&](const MtlTable &T)
    {
        if (csv_names != nullptr)
            *csv_names = T.names;
        if (csv_prop != nullptr)
        {
            csv_prop->clear();
            for (size_t c = 0; c < T.columns.size(); ++c)
            {
                std::vector<dtype> col(T.values[c].size());
                for (size_t r = 0; r < T.values[c].size(); ++r)
                    col[r] = (dtype)T.values[c][r];
                (*csv_prop)[T.columns[c]] = std::move(col);
            }
        }
    };

    // Reset all geometry / .mtl outputs to empty (shaped 0-row where applicable)
    auto empty_geometry_outputs = [&]()
    {
        if (mesh != nullptr)
            mesh->set_size(0, 9);
        if (vert_list != nullptr)
            vert_list->set_size(0, 3);
        if (face_ind != nullptr)
            face_ind->set_size(0, 3);
        if (obj_ind != nullptr)
            obj_ind->set_size(0);
        if (obj_names != nullptr)
            obj_names->clear();
        if (mtl_ind != nullptr)
            mtl_ind->set_size(0);
        if (mtl_names != nullptr)
            mtl_names->clear();
        if (bsdf != nullptr)
            bsdf->set_size(0, 17);
        if (csv_ind != nullptr)
            csv_ind->set_size(0);
    };

    // Table-only mode: no geometry, just populate the requested material-table outputs
    if (fn_obj.empty())
    {
        empty_geometry_outputs();
        if (want_csv_table)
        {
            table = load_mtl_table(fn_csv);
            fill_csv_table_out(table);
        }
        return 0;
    }

    // Open the .obj file
    std::filesystem::path obj_file{fn_obj};
    if (!std::filesystem::exists(obj_file))
        throw std::invalid_argument("Error opening file: '" + fn_obj + "' does not exist.");
    if (!std::filesystem::is_regular_file(obj_file))
        throw std::invalid_argument("Error opening file: '" + fn_obj + "' is not a regular file.");

    std::ifstream fileR{obj_file, std::ios::in};
    if (!fileR.is_open())
        throw std::invalid_argument("Error opening file: failed to open '" + fn_obj + "'.");

    // Pass 1: count vertices and faces
    arma::uword n_vert = 0, n_faces = 0;
    {
        std::string line;
        while (std::getline(fileR, line))
            if (line.length() > 2 && line[0] == 'v' && line[1] == ' ')
                ++n_vert;
            else if (line.length() > 2 && line[0] == 'f' && line[1] == ' ')
                ++n_faces;
    }

    // Empty / degenerate mesh -> empty geometry; table outputs still valid
    if (n_vert == 0 || n_faces == 0)
    {
        empty_geometry_outputs();
        if (want_csv_table)
        {
            table = load_mtl_table(fn_csv);
            fill_csv_table_out(table);
        }
        return 0;
    }

    // Count-only fast path: nothing requested
    const bool any_output = mesh || vert_list || face_ind || obj_ind || obj_names || mtl_ind || mtl_names || bsdf || want_csv_table;
    if (!any_output)
        return n_faces;

    // Load the material table now if any csv output is requested (needed for csv_ind resolution)
    if (want_csv_table)
        table = load_mtl_table(fn_csv);

    // What needs to be materialized during the parse
    const bool need_vert = (mesh != nullptr || vert_list != nullptr);
    const bool need_face = (mesh != nullptr || face_ind != nullptr);
    const bool collect_mtl = (mtl_ind != nullptr || mtl_names != nullptr || bsdf != nullptr);

    // Vertex buffer (output or scratch)
    std::vector<dtype> vert_scratch;
    dtype *p_vert = nullptr;
    if (need_vert)
    {
        if (vert_list != nullptr)
        {
            vert_list->set_size(n_vert, 3);
            p_vert = vert_list->memptr();
        }
        else
        {
            vert_scratch.resize((size_t)n_vert * 3);
            p_vert = vert_scratch.data();
        }
    }

    // Face-index buffer (output or scratch)
    std::vector<arma::uword> face_scratch;
    arma::uword *p_face = nullptr;
    if (need_face)
    {
        if (face_ind != nullptr)
        {
            face_ind->set_size(n_faces, 3);
            p_face = face_ind->memptr();
        }
        else
        {
            face_scratch.resize((size_t)n_faces * 3);
            p_face = face_scratch.data();
        }
    }

    // Per-face index outputs
    arma::uword *p_obj = nullptr;
    if (obj_ind != nullptr)
    {
        obj_ind->set_size(n_faces);
        p_obj = obj_ind->memptr();
    }
    arma::uword *p_mtl = nullptr;
    if (mtl_ind != nullptr)
    {
        mtl_ind->set_size(n_faces);
        p_mtl = mtl_ind->memptr();
    }
    arma::uword *p_csv = nullptr;
    if (csv_ind != nullptr)
    {
        csv_ind->set_size(n_faces);
        p_csv = csv_ind->memptr();
    }

    // Material / object interning helpers
    std::vector<std::string> local_mtl_names;
    std::unordered_map<std::string, arma::uword> mtl_lookup;
    std::vector<std::string> local_obj_names;

    auto intern_mtl = [&](const std::string &nm) -> arma::uword
    {
        auto it = mtl_lookup.find(nm);
        if (it != mtl_lookup.end())
            return it->second;
        arma::uword idx = (arma::uword)local_mtl_names.size();
        local_mtl_names.push_back(nm);
        mtl_lookup[nm] = idx;
        return idx;
    };

    auto resolve_csv = [&](const std::string &raw) -> arma::uword
    {
        auto it = table.index.find(raw);
        if (it != table.index.end())
            return it->second;
        std::string stripped = base_material_name(raw);
        if (stripped != raw)
        {
            auto it2 = table.index.find(stripped);
            if (it2 != table.index.end())
                return it2->second;
        }
        if (csv_strict)
            throw std::invalid_argument("Error: material '" + raw + "' referenced in '" + fn_obj + "' is not present in the material table.");
        return 0; // row 0 = transparent fallback
    };

    // Pass 2: parse geometry, objects and materials
    fileR.clear();
    fileR.seekg(0, std::ios::beg);

    arma::uword i_vert = 0, i_face = 0;
    arma::uword cur_obj = 0;
    bool have_obj = false;
    arma::uword cur_mtl = 0;
    bool mtl_set = false;
    arma::uword cur_csv = 0;
    bool simple_face_format = true;
    std::string mtllib_fn;
    std::string line;

    while (std::getline(fileR, line))
    {
        // mtllib reference (used later to locate the .mtl file)
        if (line.rfind("mtllib ", 0) == 0)
        {
            mtllib_fn = trim_ws(line.substr(7));
            continue;
        }

        // Vertex "v "
        if (line.length() > 2 && line[0] == 'v' && line[1] == ' ')
        {
            if (i_vert >= n_vert)
                throw std::invalid_argument("Error reading vertex data in '" + fn_obj + "'.");
            if (need_vert)
            {
                double x = 0.0, y = 0.0, z = 0.0;
                std::sscanf(line.c_str(), "v %lf %lf %lf", &x, &y, &z);
                p_vert[i_vert] = (dtype)x;
                p_vert[i_vert + n_vert] = (dtype)y;
                p_vert[i_vert + 2 * n_vert] = (dtype)z;
            }
            ++i_vert;
            continue;
        }

        // Face "f " (1-based indices in the file)
        if (line.length() > 2 && line[0] == 'f' && line[1] == ' ')
        {
            if (i_face >= n_faces)
                throw std::invalid_argument("Error reading face data in '" + fn_obj + "'.");

            unsigned long long a = 0, b = 0, c = 0, d = 0;
            if (simple_face_format)
            {
                std::sscanf(line.c_str(), "f %llu %llu %llu %llu", &a, &b, &c, &d);
                simple_face_format = (b != 0);
            }
            if (!simple_face_format)
                std::sscanf(line.c_str(), "f %llu%*[/0-9] %llu%*[/0-9] %llu%*[/0-9] %llu", &a, &b, &c, &d);

            if (a == 0 || b == 0 || c == 0)
                throw std::invalid_argument("Error reading face data in '" + fn_obj + "'.");
            if (d != 0)
                throw std::invalid_argument("Mesh in '" + fn_obj + "' is not triangulated (quads/n-gons unsupported).");

            // Assign material (lazily create a "default" entry for faces before any usemtl)
            if (collect_mtl && !mtl_set)
                cur_mtl = intern_mtl("default");
            if (p_mtl != nullptr)
                p_mtl[i_face] = cur_mtl;
            if (p_csv != nullptr)
                p_csv[i_face] = cur_csv;

            // Assign object (synthetic single object for files without an "o" line)
            if (!have_obj)
            {
                local_obj_names.push_back("object");
                cur_obj = 0;
                have_obj = true;
            }
            if (p_obj != nullptr)
                p_obj[i_face] = cur_obj;

            if (need_face)
            {
                p_face[i_face] = (arma::uword)a - 1;
                p_face[i_face + n_faces] = (arma::uword)b - 1;
                p_face[i_face + 2 * n_faces] = (arma::uword)c - 1;
            }
            ++i_face;
            continue;
        }

        // Object "o " (precedes the object's vertices, materials and faces)
        if (line.length() > 2 && line[0] == 'o' && line[1] == ' ')
        {
            local_obj_names.push_back(trim_ws(line.substr(2)));
            cur_obj = (arma::uword)local_obj_names.size() - 1;
            have_obj = true;
            // A new object resets the active material
            mtl_set = false;
            cur_csv = 0;
            continue;
        }

        // Material "usemtl " (defines the material for the following faces)
        if (line.rfind("usemtl ", 0) == 0)
        {
            std::string raw = trim_ws(line.substr(7));
            if (collect_mtl)
                cur_mtl = intern_mtl(raw); // raw name -> .mtl / bsdf
            mtl_set = true;
            if (csv_ind != nullptr)
                cur_csv = resolve_csv(raw); // stripped name -> EM/acoustic table
            continue;
        }
    }

    // Names outputs
    if (obj_names != nullptr)
        *obj_names = local_obj_names;
    if (mtl_names != nullptr)
        *mtl_names = local_mtl_names;

    // Assemble the triangle mesh from vertices and face indices
    if (mesh != nullptr)
    {
        mesh->set_size(n_faces, 9);
        dtype *p_mesh = mesh->memptr();
        for (arma::uword n = 0; n < n_faces; ++n)
        {
            arma::uword a = p_face[n], b = p_face[n + n_faces], c = p_face[n + 2 * n_faces];
            if (a >= n_vert || b >= n_vert || c >= n_vert)
                throw std::invalid_argument("Error assembling triangle mesh from '" + fn_obj + "'.");

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

    fileR.close();

    // EM/acoustic table outputs (csv_ind was filled during the parse)
    if (want_csv_table)
        fill_csv_table_out(table);

    // Visual BSDF from the companion .mtl file
    if (bsdf != nullptr)
        read_mtl_bsdf<dtype>(obj_file, mtllib_fn, local_mtl_names, bsdf);

    return n_faces;
}

template arma::uword quadriga_lib::obj_file_read(const std::string &fn_obj, arma::Mat<float> *mesh, arma::Mat<float> *vert_list,
                                                 arma::umat *face_ind, arma::uvec *obj_ind, std::vector<std::string> *obj_names,
                                                 arma::uvec *mtl_ind, std::vector<std::string> *mtl_names, arma::Mat<float> *bsdf,
                                                 const std::string &fn_csv, arma::uvec *csv_ind, std::vector<std::string> *csv_names,
                                                 std::unordered_map<std::string, std::vector<float>> *csv_prop, bool csv_strict);

template arma::uword quadriga_lib::obj_file_read(const std::string &fn_obj, arma::Mat<double> *mesh, arma::Mat<double> *vert_list,
                                                 arma::umat *face_ind, arma::uvec *obj_ind, std::vector<std::string> *obj_names,
                                                 arma::uvec *mtl_ind, std::vector<std::string> *mtl_names, arma::Mat<double> *bsdf,
                                                 const std::string &fn_csv, arma::uvec *csv_ind, std::vector<std::string> *csv_names,
                                                 std::unordered_map<std::string, std::vector<double>> *csv_prop, bool csv_strict);