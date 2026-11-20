// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <cstdio>
#include <climits>
#include <set>
#include <map>
#include <limits>
#include <cstdlib>
#include <type_traits>

// Built-in EM/acoustic material table (Rec. ITU-R P.2040-3; irr_glass: 3GPP TR 38.901 V17.0.0, Table 7.4.3-1).
// Stored as a CSV literal so it is parsed through the exact same path as a user-supplied CSV file.
// Faces referencing a material not present here (csv_strict == false) are assigned index 0 (no material).
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
static inline std::string qd_trim_ws(const std::string &s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
        return std::string();
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Remove a trailing '\r' (files with Windows line endings read on Linux / macOS)
static inline void qd_strip_cr(std::string &line)
{
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
}

// Remove leading spaces and tabs (indented .mtl statements)
static inline void qd_strip_leading_ws(std::string &line)
{
    size_t b = line.find_first_not_of(" \t");
    line.erase(0, b == std::string::npos ? line.size() : b);
}

// Parse the vertex references of a face line "f r1 r2 r3 ...". A reference is "v", "v/vt", "v//vn" or "v/vt/vn";
// v is a 1-based index, negative values count back from the last vertex defined before the line (OBJ relative
// indices); the texture / normal parts are skipped. Reads at most 4 references and stops at the first token that
// does not start with an integer. A reference is stored as magnitude (saturated, so any overlong value stays out of
// range) and sign. Returns the number of references read.
static inline int qd_parse_face_refs(const char *s, unsigned long long *mag, bool *neg)
{
    int n = 0;
    ++s; // 'f'
    while (n < 4)
    {
        while (*s == ' ' || *s == '\t')
            ++s;
        neg[n] = (*s == '-');
        if (*s == '-' || *s == '+')
            ++s;
        if (*s < '0' || *s > '9')
            break;
        unsigned long long v = 0;
        for (; *s >= '0' && *s <= '9'; ++s)
            v = (v < 100000000000000000ULL) ? 10ULL * v + (unsigned long long)(*s - '0') : ULLONG_MAX;
        mag[n++] = v;
        if (*s == '/') // texture / normal references
            while (*s == '/' || *s == '-' || *s == '+' || (*s >= '0' && *s <= '9'))
                ++s;
        if (*s != ' ' && *s != '\t')
            break; // end of line or a trailing token that is not a reference
    }
    return n;
}

// Split a CSV line into trimmed cells
// Note: std::getline drops truly-trailing empty fields, so callers pad rows to the header width before indexing.
static inline std::vector<std::string> qd_split_csv(const std::string &line)
{
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
        out.push_back(qd_trim_ws(cell));
    return out;
}

// Return the base material name: everything before the first '.' is the CSV lookup key.
// "itu_concrete.001" -> "itu_concrete"; "concrete.gray.001" -> "concrete".
// Names without a dot are returned unchanged. This lets one CSV material back many
// Blender sub-materials (concrete.gray, concrete.white, ... and their .001 duplicates).
static inline std::string qd_base_material_name(const std::string &s)
{
    size_t dot = s.find('.');
    return (dot == std::string::npos) ? s : s.substr(0, dot);
}

// Parse a material table from any input stream. The reader is schema-blind: it requires only a
// "name" column (the join key for .obj materials); every other column is treated as numeric.
// Empty numeric cells parse as 0.0. A non-empty cell must be one finite number in full ("1.5x", "nan" and "inf" are
// errors).
static MtlTable qd_parse_mtl_table(std::istream &in, const std::string &src)
{
    MtlTable T;
    std::string line;

    if (!std::getline(in, line))
        throw std::runtime_error("Error reading materials: '" + src + "' is empty.");

    // Strip UTF-8 BOM if present
    if (line.size() >= 3 && (unsigned char)line[0] == 0xEF &&
        (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
        line.erase(0, 3);

    std::vector<std::string> header = qd_split_csv(line);
    if (header.empty())
        throw std::runtime_error("Error reading materials: '" + src + "' has an empty header.");

    // Locate the mandatory "name" column
    arma::uword name_col = (arma::uword)header.size();
    for (arma::uword k = 0; k < (arma::uword)header.size(); ++k)
        if (header[k] == "name")
        {
            name_col = k;
            break;
        }
    if (name_col == (arma::uword)header.size())
        throw std::runtime_error("Error reading materials: '" + src + "' is missing required column 'name'.");

    // Collect numeric columns, preserving CSV order
    std::vector<arma::uword> num_pos;
    for (arma::uword k = 0; k < (arma::uword)header.size(); ++k)
        if (k != name_col)
        {
            if (header[k].empty())
                throw std::runtime_error("Error reading materials: '" + src + "' has an empty column name in the header.");
            T.columns.push_back(header[k]);
            num_pos.push_back(k);
        }
    T.values.resize(T.columns.size());

    std::unordered_set<std::string> seen;
    arma::uword line_num = 1;

    while (std::getline(in, line))
    {
        ++line_num;
        if (qd_trim_ws(line).empty())
            continue;

        std::vector<std::string> cell = qd_split_csv(line);
        if (cell.size() < header.size())
            cell.resize(header.size()); // pad trailing empties dropped by getline

        const std::string &nm = cell[name_col];
        if (nm.empty())
            throw std::runtime_error("Error reading materials: '" + src + "' line " +
                                     std::to_string(line_num) + " has an empty material name.");
        if (!seen.insert(nm).second)
            throw std::runtime_error("Error reading materials: duplicate material name '" + nm + "' in '" + src + "'.");

        T.index[nm] = (arma::uword)T.names.size();
        T.names.push_back(nm);

        for (size_t c = 0; c < T.columns.size(); ++c)
        {
            const std::string &v = cell[num_pos[c]];
            double d = 0.0; // empty cell -> 0.0
            if (!v.empty())
            {
                size_t n_read = 0;
                try
                {
                    d = std::stod(v, &n_read);
                }
                catch (const std::exception &)
                {
                    n_read = 0;
                }
                if (n_read != v.size() || !std::isfinite(d)) // the whole cell must be one finite number
                    throw std::runtime_error("Error reading materials: '" + src + "' line " +
                                             std::to_string(line_num) + " column '" + T.columns[c] +
                                             "' is not numeric ('" + v + "').");
            }
            T.values[c].push_back(d);
        }
    }

    if (T.names.empty())
        throw std::runtime_error("Error reading materials: '" + src + "' contains no material rows.");

    return T;
}

// Load the material table: built-in default if fn_csv is empty, otherwise from the CSV file.
static MtlTable qd_load_mtl_table(const std::string &fn_csv)
{
    if (fn_csv.empty())
    {
        std::istringstream ss(DEFAULT_MATERIAL_CSV);
        return qd_parse_mtl_table(ss, "<default materials>");
    }
    if (!std::filesystem::exists(fn_csv))
        throw std::runtime_error("Error opening file: CSV file '" + fn_csv + "' does not exist.");
    std::ifstream f(fn_csv);
    if (!f.is_open())
        throw std::runtime_error("Error opening file: failed to open CSV file '" + fn_csv + "'.");
    return qd_parse_mtl_table(f, fn_csv);
}

// Read Principled BSDF (visual) properties from the companion .mtl file, one row per name in 'names'
// (matched against "newmtl" by raw name). Columns: R,G,B,alpha,roughness,metallic,ior,specular,
// Re,Ge,Be,sheen,clearcoat,clearcoat_roughness,anisotropic,anisotropic_rotation,transmission.
// If the .mtl is absent or no name matches, bsdf is returned empty ([0,17]).
template <typename dtype>
static void qd_read_mtl_bsdf(const std::filesystem::path &obj_file, const std::string &mtllib_fn,
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
        throw std::runtime_error("Error opening file: failed to open '" + mtl_file.filename().string() + "'.");

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
        double alpha = 1.0;                  // Opacity, 1 = opaque
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

        // Find "newmtl <name>" (the name compared without surrounding whitespace, like the "usemtl" names)
        while (std::getline(fileR, line))
        {
            qd_strip_leading_ws(line);
            if (line.rfind("newmtl ", 0) == 0 && qd_trim_ws(line.substr(7)) == names[i_mtl])
            {
                any_found = true;
                break;
            }
        }

        // Scan until the next "newmtl " or EOF (statements may be indented)
        while (std::getline(fileR, line))
        {
            qd_strip_cr(line);
            qd_strip_leading_ws(line);
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

// Convert mesh to vert_list + face_ind (Blender-style, per-object self-contained)
// - Co-located vertices within "threshold" belonging to the SAME object are merged into one.
// - Identical coordinates in DIFFERENT objects are kept separate (duplicated); no cross-object
//  referencing, each object is self-contained.
// - Vertices are emitted object-by-object (in block order), so "vert_list" is grouped by object.
//  "face_ind" holds 0-based GLOBAL indices into "vert_list".
// - Requires the faces of each object to form a single contiguous block in "obj_ind"
//  (e.g. {1,1,2,2} is valid; {1,1,2,2,1} throws).
// - Weld uses a spatial hash grid with cell size "threshold" (27-cell neighborhood search);
//  lowest vertex index wins, which reproduces the insertion order of a greedy linear scan.
// - Objects are welded independently and in parallel (OpenMP); output ordering is unaffected.
template <typename dtype>
static void qd_mesh2vert_list(const arma::Mat<dtype> &mesh, // mesh, Size: [ n_mesh, 9 ]
                           const arma::uvec &obj_ind,    // Object index, 0-based, Size: [ n_mesh ]
                           arma::Mat<dtype> &vert_list,  // Out: List of vertices, Size: [ n_vert_out, 3 ]
                           arma::umat &face_ind,         // Out: face indices, 0-based, Size: [ n_mesh, 3 ]
                           dtype threshold)              // Co-location threshold for vertices, Default: 1 mm
{
    const arma::uword n_mesh = mesh.n_rows;

    if (mesh.n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");

    if (obj_ind.n_elem != n_mesh)
        throw std::invalid_argument("Input 'obj_ind' must have one element per mesh face.");

    if (n_mesh == 0)
    {
        vert_list.reset();
        face_ind.reset();
        return;
    }

    // Guard: faces of each object must form a single contiguous block
    {
        std::unordered_set<arma::uword> seen;
        arma::uword prev = obj_ind.at(0);
        seen.insert(prev);
        for (arma::uword n = 1; n < n_mesh; ++n)
        {
            const arma::uword cur = obj_ind.at(n);
            if (cur != prev)
            {
                if (!seen.insert(cur).second) // object re-appears after a different one
                    throw std::invalid_argument("Faces of each object must form a contiguous block in 'obj_ind'.");
                prev = cur;
            }
        }
    }

    const dtype threshold_sq = threshold * threshold;
    const dtype cell = (threshold > (dtype)0) ? threshold : (dtype)1e-6;
    const double inv_cell = 1.0 / (double)cell;

    // Object block boundaries (contiguity already verified above)
    std::vector<arma::uword> blk_start, blk_end;
    {
        arma::uword s = 0;
        for (arma::uword n = 1; n < n_mesh; ++n)
            if (obj_ind.at(n) != obj_ind.at(n - 1))
            {
                blk_start.push_back(s);
                blk_end.push_back(n);
                s = n;
            }
        blk_start.push_back(s);
        blk_end.push_back(n_mesh);
    }
    const size_t n_blk = blk_start.size();

    // Spatial hash grid key: integer cell coordinates of size "threshold"
    struct CellKey
    {
        long long x, y, z;
        bool operator==(const CellKey &o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct CellHash
    {
        size_t operator()(const CellKey &k) const
        {
            return (size_t)((unsigned long long)k.x * 73856093ULL) ^ (size_t)((unsigned long long)k.y * 19349663ULL) ^
                   (size_t)((unsigned long long)k.z * 83492791ULL);
        }
    };

    face_ind.set_size(n_mesh, 3);

    // Per-object vertex coordinates, concatenated later
    std::vector<std::vector<dtype>> bvx(n_blk), bvy(n_blk), bvz(n_blk);
    std::vector<char> non_finite(n_blk, 0); // Per object, set when a coordinate is NaN or infinite

    // Weld each object independently; objects never share vertices
#pragma omp parallel for schedule(dynamic) if (n_blk > 1 && n_mesh >= 64)
    for (long long b = 0; b < (long long)n_blk; ++b)
    {
        const size_t bb = (size_t)b;
        const arma::uword s = blk_start[bb], e = blk_end[bb];

        std::vector<dtype> &vx = bvx[bb], &vy = bvy[bb], &vz = bvz[bb];
        vx.reserve(3 * (e - s));
        vy.reserve(3 * (e - s));
        vz.reserve(3 * (e - s));

        std::unordered_map<CellKey, std::vector<arma::uword>, CellHash> grid;
        grid.reserve(3 * (e - s));

        const arma::uword NONE = std::numeric_limits<arma::uword>::max();

        for (arma::uword n = s; n < e; ++n)
            for (arma::uword k = 0; k < 3; ++k) // three triangle corners
            {
                const dtype x = mesh.at(n, 3 * k);
                const dtype y = mesh.at(n, 3 * k + 1);
                const dtype z = mesh.at(n, 3 * k + 2);

                // Cell indices; far-away cells are clamped to +/- 2^62 (the distance test still decides), which keeps
                // the casts and the neighbour indices below in range; the same comparison catches NaN and Inf
                constexpr double cell_max = 4611686018427387904.0; // 2^62
                double fx = std::floor((double)x * inv_cell), fy = std::floor((double)y * inv_cell), fz = std::floor((double)z * inv_cell);
                if (!(std::abs(fx) < cell_max && std::abs(fy) < cell_max && std::abs(fz) < cell_max))
                {
                    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                    {
                        non_finite[bb] = 1;
                        face_ind.at(n, k) = 0;
                        continue;
                    }
                    fx = std::max(-cell_max, std::min(cell_max, fx));
                    fy = std::max(-cell_max, std::min(cell_max, fy));
                    fz = std::max(-cell_max, std::min(cell_max, fz));
                }
                const long long cx = (long long)fx, cy = (long long)fy, cz = (long long)fz;

                // Search the 27 neighboring cells; lowest index wins (= greedy scan order)
                arma::uword idx = NONE;
                for (long long ix = cx - 1; ix <= cx + 1; ++ix)
                    for (long long iy = cy - 1; iy <= cy + 1; ++iy)
                        for (long long iz = cz - 1; iz <= cz + 1; ++iz)
                        {
                            auto it = grid.find(CellKey{ix, iy, iz});
                            if (it == grid.end())
                                continue;

                            for (const arma::uword g : it->second) // bucket is sorted by insertion
                            {
                                if (g >= idx)
                                    break;
                                const dtype dx = vx[g] - x, dy = vy[g] - y, dz = vz[g] - z;
                                if (dx * dx + dy * dy + dz * dz <= threshold_sq)
                                {
                                    idx = g;
                                    break;
                                }
                            }
                        }

                if (idx == NONE) // add new vertex
                {
                    idx = (arma::uword)vx.size();
                    vx.push_back(x);
                    vy.push_back(y);
                    vz.push_back(z);
                    grid[CellKey{cx, cy, cz}].push_back(idx);
                }

                face_ind.at(n, k) = idx; // object-local index, shifted to global below
            }
    }

    for (size_t b = 0; b < n_blk; ++b)
        if (non_finite[b])
            throw std::invalid_argument("Input 'mesh' must contain finite coordinates only.");

    // Prefix offsets over the per-object vertex lists
    std::vector<arma::uword> offs(n_blk + 1, 0);
    for (size_t b = 0; b < n_blk; ++b)
        offs[b + 1] = offs[b] + (arma::uword)bvx[b].size();

    // Assemble output vertex list, Size: [ n_vert, 3 ]
    const arma::uword n_vert = offs[n_blk];
    vert_list.set_size(n_vert, 3);

#pragma omp parallel for schedule(dynamic) if (n_blk > 1 && n_vert >= 8192)
    for (long long b = 0; b < (long long)n_blk; ++b)
    {
        const size_t bb = (size_t)b;
        const arma::uword o = offs[bb];

        for (arma::uword n = blk_start[bb]; n < blk_end[bb]; ++n)
            for (arma::uword k = 0; k < 3; ++k)
                face_ind.at(n, k) += o; // 0-based global index

        for (size_t i = 0; i < bvx[bb].size(); ++i)
        {
            vert_list.at(o + i, 0) = bvx[bb][i];
            vert_list.at(o + i, 1) = bvy[bb][i];
            vert_list.at(o + i, 2) = bvz[bb][i];
        }
    }
}

// Split objects into connected components ("separate by loose parts")
// - Two faces belong to the same part if they share at least one vertex index; connectivity is
//  therefore defined by the welded vertex list, not by coordinates.
// - Parts never span objects, even if objects happen to share vertex indices.
// - Union-find over vertex indices, O(n_mesh * inverse-Ackermann); parts are numbered in order of
//  their first face, so the output order follows the input face order.
// - Parts are disjoint by construction, so splitting never duplicates vertices.
static void qd_separate_loose_parts(const arma::umat &face_ind, // Face indices, Size: [ n_mesh, 3 ]
                                 const arma::uvec &obj_ind,  // Object index or empty, Size: [ n_mesh ]
                                 arma::uword n_vert,         // Number of vertices in the vertex list
                                 std::vector<arma::uword> &face_order, // Out: face indices grouped by part
                                 std::vector<arma::uword> &part_bnd,   // Out: part boundaries in "face_order", Size: [ n_part+1 ]
                                 std::vector<arma::uword> &part_obj)   // Out: parent object of each part, Size: [ n_part ]
{
    const arma::uword n_mesh = face_ind.n_rows;

    std::vector<arma::uword> parent(n_vert);
    std::vector<unsigned char> rank(n_vert, 0);
    for (arma::uword i = 0; i < n_vert; ++i)
        parent[i] = i;

    auto find = [&parent](arma::uword x) -> arma::uword
    {
        while (parent[x] != x) // path halving
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](arma::uword a, arma::uword b)
    {
        a = find(a), b = find(b);
        if (a == b)
            return;
        if (rank[a] < rank[b])
            std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b])
            ++rank[a];
    };

    for (arma::uword i = 0; i < n_mesh; ++i)
    {
        unite(face_ind.at(i, 0), face_ind.at(i, 1));
        unite(face_ind.at(i, 1), face_ind.at(i, 2));
    }

    // Assign a part to each face; key on (root, object) so parts cannot span objects
    struct PartKey
    {
        arma::uword root, obj;
        bool operator==(const PartKey &o) const { return root == o.root && obj == o.obj; }
    };
    struct PartHash
    {
        size_t operator()(const PartKey &k) const
        {
            return (size_t)(k.root * 1000003ULL) ^ (size_t)(k.obj * 2654435761ULL);
        }
    };

    std::unordered_map<PartKey, arma::uword, PartHash> key2part;
    key2part.reserve(1024);

    std::vector<arma::uword> part_of_face(n_mesh), count;
    part_obj.clear();

    for (arma::uword i = 0; i < n_mesh; ++i)
    {
        const PartKey key{find(face_ind.at(i, 0)), !obj_ind.empty() ? obj_ind.at(i) : (arma::uword)0};

        arma::uword p;
        if (auto it = key2part.find(key); it != key2part.end())
            p = it->second;
        else // new part, numbered in order of first appearance
        {
            p = (arma::uword)count.size();
            key2part.emplace(key, p);
            count.push_back(0);
            part_obj.push_back(key.obj);
        }

        part_of_face[i] = p;
        ++count[p];
    }

    // Counting sort of the faces by part (stable: face order within a part is preserved)
    const arma::uword n_part = (arma::uword)count.size();
    part_bnd.assign(n_part + 1, 0);
    for (arma::uword p = 0; p < n_part; ++p)
        part_bnd[p + 1] = part_bnd[p] + count[p];

    std::vector<arma::uword> cursor(part_bnd.begin(), part_bnd.end() - 1);
    face_order.resize(n_mesh);
    for (arma::uword i = 0; i < n_mesh; ++i)
        face_order[cursor[part_of_face[i]]++] = i;
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

- Parses a triangulated `.obj`; quads and n-gons are rejected. Face references may be `v`, `v/vt`, `v//vn` or
  `v/vt/vn`; a negative (relative) index counts back from the last vertex defined before the face line (`-1` = that
  vertex). Index 0, or an index beyond the vertices defined before the face, throws `std::runtime_error`. The file is
  parsed and validated for any combination of requested outputs, also when no output is requested.
- Lines may end in LF or CRLF (`.obj`, `.mtl` and material CSV).
- Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`). Unmatched names
  throw when `csv_strict = true`; otherwise they map to index 0 (no material). The two index spaces are
  decoupled, so several visual materials (e.g. `wall.001`, `wall.002`) may resolve to a single EM material.
- `usemtl (null)` (written by [[obj_file_write]]) marks the following faces as having no material: index 0 in `mtl_ind`
  and `csv_ind` (also with `csv_strict`); `(null)` is not listed in `mtl_names` or `bsdf`.
- Geometry indices (`face_ind`, `obj_ind`) are 0-based. Material indices (`mtl_ind`, `csv_ind`) are
  1-based, with 0 reserved for the outside / no-material state (faces with no assigned material;
  for `csv_ind` in non-strict mode, also materials absent from the table).
- With an empty `fn`, geometry and `.mtl` outputs are empty and only the table (`csv_names`, `csv_prop`) 
  is populated — useful for inspecting a CSV or the default library. If `fn_csv` is also empty, the built-in default table is returned.
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>
  
## Declaration:
```
arma::uword quadriga_lib::obj_file_read(
    const std::string &fn,
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
- **`fn`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** — Path to an EM/acoustic material CSV; must contain a `name` column. Unmatched faces map
  to index 0 (no material) unless `csv_strict` is set. Empty uses the built-in ITU-R P.2040 default table.
- **`csv_strict`** — If `true`, throw when a `usemtl` material is absent from the table; otherwise map to index 0 (no material)

## Outputs:
- **`mesh`** — Triangle vertex coordinates `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per row; `[n_mesh, 9]`
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 0-based vertex indices into `vert_list` per triangle; `[n_mesh, 3]`
- **`obj_ind`** — 0-based object index per triangle; `[n_mesh]`
- **`obj_names`** — Object names; length `max(obj_ind)+1`
- **`mtl_ind`** — 1-based visual-material index per triangle (0 = no material); `[n_mesh]`
- **`mtl_names`** — Visual material names (raw `usemtl`); length `n_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `[n_mtl, 17]`
- **`csv_ind`** — 1-based EM/acoustic-material index per triangle (0 = no material); `[n_mesh]`
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
arma::uword quadriga_lib::obj_file_read(const std::string &fn, arma::Mat<dtype> *mesh, arma::Mat<dtype> *vert_list,
                                        arma::umat *face_ind, arma::uvec *obj_ind, std::vector<std::string> *obj_names,
                                        arma::uvec *mtl_ind, std::vector<std::string> *mtl_names, arma::Mat<dtype> *bsdf,
                                        const std::string &fn_csv, arma::uvec *csv_ind, std::vector<std::string> *csv_names,
                                        std::unordered_map<std::string, std::vector<dtype>> *csv_prop, bool csv_strict)
{
    qd_check_alias_outputs(mesh, "mesh", vert_list, "vert_list");
    qd_check_alias_outputs(mesh, "mesh", bsdf, "bsdf");
    qd_check_alias_outputs(vert_list, "vert_list", bsdf, "bsdf");
    qd_check_alias_outputs(face_ind, "face_ind", obj_ind, "obj_ind");
    qd_check_alias_outputs(face_ind, "face_ind", mtl_ind, "mtl_ind");
    qd_check_alias_outputs(face_ind, "face_ind", csv_ind, "csv_ind");
    qd_check_alias_outputs(obj_ind, "obj_ind", mtl_ind, "mtl_ind");
    qd_check_alias_outputs(obj_ind, "obj_ind", csv_ind, "csv_ind");
    qd_check_alias_outputs(mtl_ind, "mtl_ind", csv_ind, "csv_ind");
    qd_check_alias_outputs(obj_names, "obj_names", mtl_names, "mtl_names");
    qd_check_alias_outputs(obj_names, "obj_names", csv_names, "csv_names");
    qd_check_alias_outputs(mtl_names, "mtl_names", csv_names, "csv_names");

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
    if (fn.empty())
    {
        empty_geometry_outputs();
        if (want_csv_table)
        {
            table = qd_load_mtl_table(fn_csv);
            fill_csv_table_out(table);
        }
        return 0;
    }

    // Open the .obj file
    std::filesystem::path obj_file{fn};
    if (!std::filesystem::exists(obj_file))
        throw std::runtime_error("Error opening file: '" + fn + "' does not exist.");
    if (!std::filesystem::is_regular_file(obj_file))
        throw std::runtime_error("Error opening file: '" + fn + "' is not a regular file.");

    std::ifstream fileR{obj_file, std::ios::in};
    if (!fileR.is_open())
        throw std::runtime_error("Error opening file: failed to open '" + fn + "'.");

    // Pass 1: count vertices and faces
    arma::uword n_vert = 0, n_faces = 0;
    {
        std::string line;
        while (std::getline(fileR, line))
        {
            qd_strip_cr(line);
            if (line.length() > 2 && line[0] == 'v' && line[1] == ' ')
                ++n_vert;
            else if (line.length() > 2 && line[0] == 'f' && line[1] == ' ')
                ++n_faces;
        }
    }

    // No faces -> empty geometry; table outputs still valid (faces without vertices fail in the parse below)
    if (n_faces == 0)
    {
        empty_geometry_outputs();
        if (want_csv_table)
        {
            table = qd_load_mtl_table(fn_csv);
            fill_csv_table_out(table);
        }
        return 0;
    }

    // Without outputs the parse below still runs: it validates the faces (index range, triangles)

    // Load the material table now if any csv output is requested (needed for csv_ind resolution)
    if (want_csv_table)
        table = qd_load_mtl_table(fn_csv);

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
            return it->second + 1;
        std::string stripped = qd_base_material_name(raw);
        if (stripped != raw)
        {
            auto it2 = table.index.find(stripped);
            if (it2 != table.index.end())
                return it2->second + 1;
        }
        if (csv_strict)
            throw std::runtime_error("Material '" + raw + "' referenced in '" + fn + "' is not present in the material table.");
        return 0; // 0 = no material (outside)
    };

    // Pass 2: parse geometry, objects and materials
    fileR.clear();
    fileR.seekg(0, std::ios::beg);

    arma::uword i_vert = 0, i_face = 0;
    arma::uword cur_obj = 0;
    bool have_obj = false;
    arma::uword cur_mtl = 0;
    arma::uword cur_csv = 0;
    std::string mtllib_fn;
    std::string line;

    while (std::getline(fileR, line))
    {
        qd_strip_cr(line);

        // mtllib reference (used later to locate the .mtl file)
        if (line.rfind("mtllib ", 0) == 0)
        {
            mtllib_fn = qd_trim_ws(line.substr(7));
            continue;
        }

        // Vertex "v "
        if (line.length() > 2 && line[0] == 'v' && line[1] == ' ')
        {
            if (i_vert >= n_vert)
                throw std::runtime_error("Error reading vertex data in '" + fn + "'.");
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

        // Face "f " (1-based or negative relative indices in the file)
        if (line.length() > 2 && line[0] == 'f' && line[1] == ' ')
        {
            if (i_face >= n_faces)
                throw std::runtime_error("Error reading face data in '" + fn + "'.");

            unsigned long long mag[4];
            bool neg[4];
            const int n_ref = qd_parse_face_refs(line.c_str(), mag, neg);
            if (n_ref < 3)
                throw std::runtime_error("Error reading face data in '" + fn + "'.");
            if (n_ref > 3)
                throw std::runtime_error("Mesh in '" + fn + "' is not triangulated (quads/n-gons unsupported).");

            // 0-based vertex index; index 0 and indices beyond the vertices defined so far are invalid
            arma::uword abc[3];
            for (int k = 0; k < 3; ++k)
            {
                if (mag[k] == 0 || mag[k] > (unsigned long long)i_vert)
                    throw std::runtime_error("Error reading face data in '" + fn + "': face " + std::to_string(i_face + 1) +
                                             " references a vertex that is 0 or beyond the " + std::to_string(i_vert) +
                                             " vertices defined before it.");
                abc[k] = neg[k] ? i_vert - (arma::uword)mag[k] : (arma::uword)mag[k] - 1;
            }

            // Assign material (lazily create a "default" entry for faces before any usemtl)
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
                p_face[i_face] = abc[0];
                p_face[i_face + n_faces] = abc[1];
                p_face[i_face + 2 * n_faces] = abc[2];
            }
            ++i_face;
            continue;
        }

        // Object "o " (precedes the object's vertices, materials and faces)
        if (line.length() > 2 && line[0] == 'o' && line[1] == ' ')
        {
            local_obj_names.push_back(qd_trim_ws(line.substr(2)));
            cur_obj = (arma::uword)local_obj_names.size() - 1;
            have_obj = true;
            // A new object resets the active material
            cur_mtl = 0;
            cur_csv = 0;
            continue;
        }

        // Material "usemtl " (defines the material for the following faces)
        // "usemtl (null)" (written by obj_file_write) marks the following faces as having no material
        if (line.rfind("usemtl ", 0) == 0)
        {
            std::string raw = qd_trim_ws(line.substr(7));
            if (raw == "(null)")
            {
                cur_mtl = 0;
                cur_csv = 0;
                continue;
            }
            if (collect_mtl)
                cur_mtl = intern_mtl(raw) + 1; // 1-based; 0 = no material
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
            arma::uword a = p_face[n], b = p_face[n + n_faces], c = p_face[n + 2 * n_faces]; // checked in the parse
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
        qd_read_mtl_bsdf<dtype>(obj_file, mtllib_fn, local_mtl_names, bsdf);

    return n_faces;
}

template arma::uword quadriga_lib::obj_file_read(const std::string &fn, arma::Mat<float> *mesh, arma::Mat<float> *vert_list,
                                                 arma::umat *face_ind, arma::uvec *obj_ind, std::vector<std::string> *obj_names,
                                                 arma::uvec *mtl_ind, std::vector<std::string> *mtl_names, arma::Mat<float> *bsdf,
                                                 const std::string &fn_csv, arma::uvec *csv_ind, std::vector<std::string> *csv_names,
                                                 std::unordered_map<std::string, std::vector<float>> *csv_prop, bool csv_strict);

template arma::uword quadriga_lib::obj_file_read(const std::string &fn, arma::Mat<double> *mesh, arma::Mat<double> *vert_list,
                                                 arma::umat *face_ind, arma::uvec *obj_ind, std::vector<std::string> *obj_names,
                                                 arma::uvec *mtl_ind, std::vector<std::string> *mtl_names, arma::Mat<double> *bsdf,
                                                 const std::string &fn_csv, arma::uvec *csv_ind, std::vector<std::string> *csv_names,
                                                 std::unordered_map<std::string, std::vector<double>> *csv_prop, bool csv_strict);

/*!MD
# obj_file_write
Write a triangulated Wavefront .obj (and .mtl) file

- Supply geometry as either `mesh`, or as `vert_list` + `face_ind`; giving both, or neither, is an error
- With `mesh`: `vert_list_out` + `face_ind_out` are derived from it, merging vertices of the same object that
  are closer than `threshold` (no merging across objects). With `vert_list`/`face_ind`: data is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- Without `obj_ind`/`obj_names`: a single object named `object` is written
- With `split_loose_parts`: objects are separated into connected components ("separate by loose parts"); connectivity
  follows the welded vertex list, so unwelded input geometry yields one part per face
- Without `mtl_ind`: no `usemtl` tags and no `.mtl` file are written. With `mtl_ind`, each face carries a
  1-based material index (0 = no material, leaving that face unassigned); leave `mtl_ind` empty to omit materials entirely
- A face with `mtl_ind = 0` that follows a face with a material in the same object (or part) is preceded by
  `usemtl (null)`, which [[obj_file_read]] reads back as 0; the `.mtl` gets no `(null)` entry. Face order and objects
  are unchanged. A material named `(null)` that is used by a face is therefore rejected (`std::invalid_argument`);
  unused entries of `mtl_names` may carry the name
- The `.mtl` (named after the `.obj`) lists each used material; values default to a gray material when `bsdf` is omitted
- Duplicate entries in `mtl_names` are merged into one `.mtl` entry if their `bsdf` rows are identical (or if `bsdf`
  is omitted); duplicates with differing `bsdf` rows are disambiguated as `name.001`, `name.002`, ...
- If `csv_names` is given, the EM/acoustic material table is written to a companion `.csv` (named after the `.obj`):
  columns follow a fixed canonical order, then any extra `csv_prop` columns (alphabetical); `csv_write_defaults`
  additionally emits canonical columns absent from `csv_prop`, filled with their defaults (`a`, `e`, `fRef` = 1,
  `side` = 3.29, else 0)

## Declaration:
```
void obj_file_write(
    const std::string &fn,
    const arma::Mat<dtype> &mesh = arma::Mat<dtype>(),
    const arma::uvec &obj_ind = arma::uvec(),
    const arma::uvec &mtl_ind = arma::uvec(),
    const std::vector<std::string> &obj_names = std::vector<std::string>(),
    const std::vector<std::string> &mtl_names = std::vector<std::string>(),
    arma::Mat<dtype> *vert_list_out = nullptr,
    arma::umat *face_ind_out = nullptr,
    const arma::Mat<dtype> &vert_list = arma::Mat<dtype>(),
    const arma::umat &face_ind = arma::umat(),
    const arma::Mat<dtype> &bsdf = arma::Mat<dtype>(),
    double threshold = 0.001,
    const arma::uvec &csv_ind = arma::uvec(),
    const std::vector<std::string> &csv_names = std::vector<std::string>(),
    const std::unordered_map<std::string, std::vector<dtype>> &csv_prop = std::unordered_map<std::string, std::vector<dtype>>(),
    bool csv_write_defaults = false,
    bool split_loose_parts = false);
```

## Inputs:
- **`fn`** — Output path; must end in `.obj`; if empty, no files are written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{x1,y1,z1,...,x3,y3,z3}` per row; `[n_mesh, 9]`; mutually exclusive with `vert_list`/`face_ind`
- **`obj_ind`** — 0-based object index per face; `[n_mesh]`; each object must be a contiguous block
- **`mtl_ind`** — 1-based material index per face (0 = no material); `[n_mesh]`; empty for no materials
- **`obj_names`** — Object names; length > `max(obj_ind)`; required if `obj_ind` is given
- **`mtl_names`** — Material names; length ≥ `max(mtl_ind)` (1-based); required if `max(mtl_ind)` > 0
- **`vert_list`** — Vertex positions; `[n_vert, 3]`; only with `face_ind`, written unchanged
- **`face_ind`** — 0-based vertex indices per face; `[n_mesh, 3]`; required with `vert_list`
- **`bsdf`** — Principled BSDF for the `.mtl`; `[n_mtl, 17]`; see [[obj_file_read]] for columns
- **`threshold`** — Vertex co-location distance for merging within an object; default 1 mm
- **`csv_ind`** — 1-based EM/acoustic-material index per face (0 = no material); `[n_mesh]`; optional, validated if given
- **`csv_names`** — EM/acoustic material names (the full table); writing the `.csv` requires this
- **`csv_prop`** — Material properties keyed by column name; each vector must have one value per `csv_names` entry
- **`csv_write_defaults`** — If `true`, also write canonical columns absent from `csv_prop`, using their defaults
- **`split_loose_parts`** — If `true`, split each object into connected components (faces sharing a vertex); parts of a split object are named `name.001`, `name.002`, ...

## Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `[n_vert, 3]`
- **`face_ind_out`** — 0-based face indices derived from `mesh`, or a copy of `face_ind`; `[n_mesh, 3]`

## See also:
- [[obj_file_read]] (for reading OBJ files and the BSDF column layout)
- [[mitsuba_xml_file_write]] (for exporting to Mitsuba scene file format)
MD!*/

template <typename dtype>
void quadriga_lib::obj_file_write(const std::string &fn,
                                  const arma::Mat<dtype> &mesh,
                                  const arma::uvec &obj_ind,
                                  const arma::uvec &mtl_ind,
                                  const std::vector<std::string> &obj_names,
                                  const std::vector<std::string> &mtl_names,
                                  arma::Mat<dtype> *vert_list_out,
                                  arma::umat *face_ind_out,
                                  const arma::Mat<dtype> &vert_list,
                                  const arma::umat &face_ind,
                                  const arma::Mat<dtype> &bsdf,
                                  double threshold_in,
                                  const arma::uvec &csv_ind,
                                  const std::vector<std::string> &csv_names,
                                  const std::unordered_map<std::string, std::vector<dtype>> &csv_prop,
                                  bool csv_write_defaults,
                                  bool split_loose_parts)
{
    const dtype threshold = dtype(threshold_in);
    // Mode selection: mesh XOR (vert_list + face_ind)
    const bool has_mesh = !mesh.empty();
    const bool has_vl = !vert_list.empty();
    const bool has_fi = !face_ind.empty();

    if (has_mesh && (has_vl || has_fi))
        throw std::invalid_argument("Provide either 'mesh' or 'vert_list'+'face_ind', not both.");
    if (!has_mesh && !has_vl && !has_fi)
        throw std::invalid_argument("Either 'mesh' or 'vert_list'+'face_ind' must be given.");
    if (!has_mesh && (!has_vl || !has_fi))
        throw std::invalid_argument("'vert_list' and 'face_ind' must be given together.");
    qd_check_alias(vert_list_out, "vert_list_out", &mesh, &vert_list, &bsdf);
    qd_check_alias(face_ind_out, "face_ind_out", &obj_ind, &mtl_ind, &face_ind, &csv_ind);

    if (fn.empty() && vert_list_out == nullptr && face_ind_out == nullptr)
        return;

    // Number of faces + basic shape checks
    arma::uword n_mesh = 0;
    if (has_mesh)
    {
        if (mesh.n_cols != 9)
            throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
        n_mesh = mesh.n_rows;
    }
    else
    {
        if (vert_list.n_cols != 3)
            throw std::invalid_argument("Input 'vert_list' must have 3 columns.");
        if (face_ind.n_cols != 3)
            throw std::invalid_argument("Input 'face_ind' must have 3 columns.");
        n_mesh = face_ind.n_rows;
    }

    if (n_mesh == 0)
        throw std::invalid_argument("No faces to write (empty geometry).");

    // Validate obj_ind: 0-based, each object a contiguous block
    if (!obj_ind.empty())
    {
        if (obj_ind.n_elem != n_mesh)
            throw std::invalid_argument("Input 'obj_ind' must have one element per face.");

        std::unordered_set<arma::uword> seen;
        arma::uword prev = obj_ind.at(0);
        seen.insert(prev);
        for (arma::uword n = 1; n < n_mesh; ++n)
        {
            const arma::uword cur = obj_ind.at(n);
            if (cur != prev)
            {
                if (!seen.insert(cur).second)
                    throw std::invalid_argument("Faces of each object must form a contiguous block in 'obj_ind'.");
                prev = cur;
            }
        }

        if (obj_names.size() <= obj_ind.max())
            throw std::invalid_argument("'obj_names' is missing or too short for the given 'obj_ind'.");
    }

    // Validate mtl_ind
    if (!mtl_ind.empty())
    {
        if (mtl_ind.n_elem != n_mesh)
            throw std::invalid_argument("Input 'mtl_ind' must have one element per face.");

        const arma::uword n_mtl = mtl_ind.max(); // mtl_ind is 1-based; 0 = no material
        if (mtl_names.size() < n_mtl)
            throw std::invalid_argument("'mtl_names' is missing or too short for the given 'mtl_ind'.");
        if (!bsdf.empty() && (bsdf.n_cols != 17 || bsdf.n_rows < n_mtl))
            throw std::invalid_argument("Input 'bsdf' must have 17 columns and one row per material.");

        // "(null)" marks faces without material in the written file (see below); a used material cannot carry it
        for (arma::uword i = 0; i < n_mtl; ++i)
            if (mtl_names[i] == "(null)" && arma::any(mtl_ind == i + 1))
                throw std::invalid_argument("Material name '(null)' is reserved for faces without material; rename it.");
    }
    else if (!bsdf.empty())
        throw std::invalid_argument("'bsdf' requires 'mtl_ind' and 'mtl_names'.");

    // Validate CSV material-table inputs (EM/acoustic side)
    const bool want_csv = (!csv_ind.empty() || !csv_names.empty() || !csv_prop.empty());
    if (want_csv)
    {
        if (csv_names.empty())
            throw std::invalid_argument("Writing the material CSV requires a non-empty 'csv_names'.");

        const arma::uword n_csv = (arma::uword)csv_names.size();

        if (!csv_prop.empty())
            for (const auto &kv : csv_prop)
                if ((arma::uword)kv.second.size() != n_csv)
                    throw std::invalid_argument("Column '" + kv.first + "' in 'csv_prop' must have one value per entry in 'csv_names'.");

        if (!csv_ind.empty())
        {
            if (csv_ind.n_elem != n_mesh)
                throw std::invalid_argument("Input 'csv_ind' must have one element per face.");
            if (csv_ind.max() > n_csv) // csv_ind is 1-based; 0 = no material
                throw std::invalid_argument("Input 'csv_ind' references a material outside 'csv_names'.");
        }
    }

    // Build / reference the indexed geometry
    arma::Mat<dtype> VL_local;
    arma::umat FI_local;
    const arma::Mat<dtype> *pVL = nullptr;
    const arma::umat *pFI = nullptr;

    if (has_mesh)
    {
        qd_mesh2vert_list(mesh, obj_ind.empty() ? arma::uvec(n_mesh, arma::fill::zeros) : obj_ind, VL_local, FI_local, threshold);
        pVL = &VL_local;
        pFI = &FI_local;
    }
    else
    {
        if (face_ind.max() >= vert_list.n_rows)
            throw std::invalid_argument("Input 'face_ind' references a vertex outside 'vert_list'.");
        pVL = &vert_list;
        pFI = &face_ind;
    }

    // Fill optional outputs
    if (vert_list_out != nullptr)
        *vert_list_out = *pVL;
    if (face_ind_out != nullptr)
        *face_ind_out = *pFI;

    // Empty filename: outputs only, write nothing
    if (fn.empty())
        return;

    // File name / paths
    auto ends_with = [](const std::string &s, const std::string &suf)
    { return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0; };

    if (!ends_with(fn, ".obj"))
        throw std::invalid_argument("Output file name must end with '.obj'.");

    std::filesystem::path obj_path(fn);
    std::filesystem::path mtl_path = obj_path;
    mtl_path.replace_extension(".mtl");
    const std::string mtllib_name = mtl_path.filename().string();

    // Per-face accessors
    constexpr arma::uword NO_MTL = std::numeric_limits<arma::uword>::max();

    auto objid = [&](arma::uword f) -> arma::uword
    { return !obj_ind.empty() ? obj_ind.at(f) : (arma::uword)0; };

    auto mtlid = [&](arma::uword f) -> arma::uword
    {
        if (mtl_ind.empty())
            return NO_MTL;
        const arma::uword v = mtl_ind.at(f);
        return (v == 0) ? NO_MTL : v - 1; // 1-based -> 0-based; 0 = no material
    };

    // Materials actually used (nonzero, sorted, unique)
    std::set<arma::uword> used_mtl;
    if (!mtl_ind.empty())
        for (arma::uword f = 0; f < n_mesh; ++f)
            if (const arma::uword m = mtlid(f); m != NO_MTL)
                used_mtl.insert(m); // 0-based material row (no-material faces skipped)
    const bool write_materials = !used_mtl.empty();

    // Resolve the material names written to the .obj / .mtl.
    // 'mtl_names' may contain duplicates: entries sharing a name are merged into a single .mtl entry
    // when their BSDF rows are identical (or when no BSDF is given). Entries sharing a name but
    // carrying different BSDF data are disambiguated with ".001", ".002", ... suffixes.
    std::unordered_map<arma::uword, std::string> mtl_out_name; // used material row -> written name
    std::vector<arma::uword> mtl_write_order;                  // representative rows, ascending

    if (write_materials)
    {
        auto same_bsdf = [&](arma::uword p, arma::uword q) -> bool
        {
            if (bsdf.empty())
                return true; // no BSDF data -> all same-named entries collapse into one
            for (arma::uword c = 0; c < 17; ++c)
                if (bsdf.at(p, c) != bsdf.at(q, c))
                    return false;
            return true;
        };

        // Group the used material rows by name (used_mtl is sorted, so rows stay ascending)
        std::map<std::string, std::vector<arma::uword>> by_name;
        for (const arma::uword id : used_mtl)
            by_name[mtl_names[id]].push_back(id);

        // Reserve every base name so generated suffixes cannot collide with an existing name
        std::unordered_set<std::string> taken;
        for (const auto &kv : by_name)
            taken.insert(kv.first);

        for (const auto &kv : by_name)
        {
            const std::string &base = kv.first;

            // Split the group into distinct BSDF variants, in order of first appearance
            std::vector<std::vector<arma::uword>> variants;
            for (const arma::uword id : kv.second)
            {
                bool placed = false;
                for (auto &v : variants)
                    if (same_bsdf(v.front(), id))
                    {
                        v.push_back(id);
                        placed = true;
                        break;
                    }
                if (!placed)
                    variants.push_back(std::vector<arma::uword>{id});
            }

            const bool need_suffix = (variants.size() > 1);
            int ctr = 0;
            for (const auto &v : variants)
            {
                std::string name = base;
                if (need_suffix)
                    do
                    {
                        char buf[16];
                        std::snprintf(buf, sizeof(buf), ".%03d", ++ctr);
                        name = base + buf;
                    } while (taken.find(name) != taken.end());
                taken.insert(name);

                for (const arma::uword id : v)
                    mtl_out_name[id] = name;
                mtl_write_order.push_back(v.front());
            }
        }

        std::sort(mtl_write_order.begin(), mtl_write_order.end()); // keep .mtl in material-row order
    }

    // Shortest round-trip number formatter (also maps -0 -> 0)
    // snprintf-based to avoid the std::to_chars float overloads (GLIBCXX_3.4.29 / GCC 11)
    auto fmt = [](dtype v) -> std::string
    {
        if (v == (dtype)0)
            return std::string("0");
        constexpr int max_prec = std::is_same<dtype, float>::value ? 9 : 17;
        char buf[64];
        for (int prec = 1; prec < max_prec; ++prec)
        {
            std::snprintf(buf, sizeof(buf), "%.*g", prec, (double)v);
            if ((dtype)std::strtod(buf, nullptr) == v)
                return std::string(buf);
        }
        std::snprintf(buf, sizeof(buf), "%.*g", max_prec, (double)v);
        return std::string(buf);
    };

    // Write .obj (per-object self-contained vertex blocks)
    std::ofstream obj(obj_path, std::ios::out | std::ios::trunc);
    if (!obj.is_open())
        throw std::runtime_error("Error opening file: failed to open '" + fn + "'.");

    obj << "# Wavefront OBJ file written by quadriga-lib\n";
    if (write_materials)
        obj << "mtllib " << mtllib_name << "\n";

    const arma::Mat<dtype> &VL = *pVL;
    const arma::umat &FI = *pFI;

    // Blocks to write: "blk_bnd" delimits them in "face_order", "blk_obj" names their parent object.
    // Without splitting this is just the object blocks in input order (face_order = identity).
    std::vector<arma::uword> face_order, blk_bnd, blk_obj;

    if (split_loose_parts)
        qd_separate_loose_parts(FI, obj_ind, VL.n_rows, face_order, blk_bnd, blk_obj);
    else
    {
        face_order.resize(n_mesh);
        for (arma::uword i = 0; i < n_mesh; ++i)
            face_order[i] = i;

        blk_bnd.push_back(0);
        for (arma::uword i = 1; i < n_mesh; ++i)
            if (objid(i) != objid(i - 1))
            {
                blk_obj.push_back(objid(i - 1));
                blk_bnd.push_back(i);
            }
        blk_obj.push_back(objid(n_mesh - 1));
        blk_bnd.push_back(n_mesh);
    }
    const size_t n_blk = blk_obj.size();

    // Parts per parent object; only objects that actually split get suffixed names
    std::unordered_map<arma::uword, arma::uword> n_parts, part_ctr;
    if (split_loose_parts)
        for (const arma::uword o : blk_obj)
            ++n_parts[o];

    auto base_name = [&](arma::uword o) -> std::string
    {
        if (!obj_ind.empty())
            return (obj_names)[o];
        if (!obj_names.empty())
            return (obj_names)[0];
        return std::string("object");
    };

    // Reserve every base name so generated suffixes cannot collide with an existing object name
    std::unordered_set<std::string> obj_taken;
    if (split_loose_parts)
        for (const arma::uword o : blk_obj)
            obj_taken.insert(base_name(o));

    arma::uword offset = 0; // cumulative vertices already written (global 1-based base)
    for (size_t blk = 0; blk < n_blk; ++blk)
    {
        const arma::uword bf = blk_bnd[blk], bg = blk_bnd[blk + 1];
        const arma::uword cur_obj = blk_obj[blk];

        // Object header; parts of a split object are suffixed ".001", ".002", ...
        std::string oname = base_name(cur_obj);
        if (split_loose_parts && n_parts[cur_obj] > 1)
        {
            const std::string base = oname;
            arma::uword &ctr = part_ctr[cur_obj];
            do
            {
                char buf[16];
                std::snprintf(buf, sizeof(buf), ".%03d", (int)++ctr);
                oname = base + buf;
            } while (obj_taken.find(oname) != obj_taken.end());
            obj_taken.insert(oname);
        }
        obj << "o " << oname << "\n";

        // Collect this block's vertices in first-use order
        std::unordered_map<arma::uword, arma::uword> remap;
        std::vector<arma::uword> order;
        order.reserve((bg - bf) * 3);
        for (arma::uword ii = bf; ii < bg; ++ii)
            for (arma::uword k = 0; k < 3; ++k)
            {
                const arma::uword gv = FI.at(face_order[ii], k);
                if (remap.find(gv) == remap.end())
                {
                    remap.emplace(gv, (arma::uword)order.size());
                    order.push_back(gv);
                }
            }

        // Vertices
        for (const arma::uword gv : order)
            obj << "v " << fmt(VL.at(gv, 0)) << " " << fmt(VL.at(gv, 1)) << " " << fmt(VL.at(gv, 2)) << "\n";

        // Faces; emit usemtl on material change (reset per object, like the reader)
        // Compared by written name, so merged duplicates do not emit a redundant tag
        // A face without material after one with material gets "usemtl (null)", which the reader maps back to 0
        std::string last_mtl_name;
        bool have_last_mtl = false;
        for (arma::uword ii = bf; ii < bg; ++ii)
        {
            const arma::uword i = face_order[ii];

            if (write_materials)
            {
                if (const arma::uword m = mtlid(i); m != NO_MTL)
                {
                    const std::string &mname = mtl_out_name.at(m);
                    if (!have_last_mtl || mname != last_mtl_name)
                    {
                        obj << "usemtl " << mname << "\n";
                        last_mtl_name = mname;
                        have_last_mtl = true;
                    }
                }
                else if (have_last_mtl)
                {
                    obj << "usemtl (null)\n";
                    have_last_mtl = false;
                }
            }

            const arma::uword a = offset + remap[FI.at(i, 0)] + 1;
            const arma::uword b = offset + remap[FI.at(i, 1)] + 1;
            const arma::uword c = offset + remap[FI.at(i, 2)] + 1;
            obj << "f " << a << " " << b << " " << c << "\n";
        }

        offset += (arma::uword)order.size();
    }
    obj.close();

    // Write .mtl (one newmtl per used material; defaults omitted)
    if (write_materials)
    {
        std::ofstream mtl(mtl_path, std::ios::out | std::ios::trunc);
        if (!mtl.is_open())
            throw std::runtime_error("Error opening file: failed to open '" + mtl_path.string() + "'.");

        mtl << "# Wavefront MTL file written by quadriga-lib\n";

        for (const arma::uword id : mtl_write_order)
        {
            mtl << "\nnewmtl " << mtl_out_name.at(id) << "\n";

            if (!bsdf.empty())
            {
                const arma::uword r = id; // id is already the 0-based material row
                const dtype R = bsdf.at(r, 0), G = bsdf.at(r, 1), B = bsdf.at(r, 2);
                const dtype d = bsdf.at(r, 3), Pr = bsdf.at(r, 4), Pm = bsdf.at(r, 5);
                const dtype Ni = bsdf.at(r, 6), Ks = bsdf.at(r, 7);
                const dtype Re = bsdf.at(r, 8), Ge = bsdf.at(r, 9), Be = bsdf.at(r, 10);
                const dtype Ps = bsdf.at(r, 11), Pc = bsdf.at(r, 12), Pcr = bsdf.at(r, 13);
                const dtype an = bsdf.at(r, 14), anr = bsdf.at(r, 15), Tf = bsdf.at(r, 16);

                if (R != (dtype)0.8 || G != (dtype)0.8 || B != (dtype)0.8) // base color
                    mtl << "Kd " << fmt(R) << " " << fmt(G) << " " << fmt(B) << "\n";
                if (d != (dtype)1.0) // dissolve (opacity)
                    mtl << "d " << fmt(d) << "\n";
                if (Pr != (dtype)0.5) // roughness (Pr, never Ns)
                    mtl << "Pr " << fmt(Pr) << "\n";
                if (Pm != (dtype)0.0) // metallic (Pm, never Ka)
                    mtl << "Pm " << fmt(Pm) << "\n";
                if (Ni != (dtype)1.45) // index of refraction
                    mtl << "Ni " << fmt(Ni) << "\n";
                if (Ks != (dtype)0.5) // specular (3 comps for Blender; reader reads first)
                    mtl << "Ks " << fmt(Ks) << " " << fmt(Ks) << " " << fmt(Ks) << "\n";
                if (Re != (dtype)0.0 || Ge != (dtype)0.0 || Be != (dtype)0.0) // emission
                    mtl << "Ke " << fmt(Re) << " " << fmt(Ge) << " " << fmt(Be) << "\n";
                if (Ps != (dtype)0.0)
                    mtl << "Ps " << fmt(Ps) << "\n";
                if (Pc != (dtype)0.0)
                    mtl << "Pc " << fmt(Pc) << "\n";
                if (Pcr != (dtype)0.0)
                    mtl << "Pcr " << fmt(Pcr) << "\n";
                if (an != (dtype)0.0)
                    mtl << "aniso " << fmt(an) << "\n";
                if (anr != (dtype)0.0)
                    mtl << "anisor " << fmt(anr) << "\n";
                if (Tf != (dtype)0.0) // transmission (3 comps; reader reads first)
                    mtl << "Tf " << fmt(Tf) << " " << fmt(Tf) << " " << fmt(Tf) << "\n";
            }
            // No bsdf -> all properties default (reader fills gray 0.8 etc.)
        }
        mtl.close();
    }

    // Write the EM/acoustic material table to a companion .csv
    if (want_csv)
    {
        static const std::vector<std::string> kOrder = {
            "a", "b", "c", "d", "e", "f", "g", "h", "thk",
            "att", "attB", "alpha", "alphaB", "fRef", "m",
            "resF", "resQ", "resS", "coiF", "coiQ", "tf", "tfB", "tfP", "tfF", "tfQ", "side"};

        auto default_for = [](const std::string &col) -> dtype
        {
            if (col == "a" || col == "e" || col == "fRef")
                return (dtype)1.0;
            if (col == "side") // the panel size a mass-law material takes where the column is absent
                return (dtype)3.29;
            return (dtype)0.0;
        };

        const arma::uword n_csv = (arma::uword)csv_names.size();

        // Output columns: canonical order first (present, or all if csv_write_defaults), then extra csv_prop columns
        std::vector<std::string> columns;
        std::unordered_set<std::string> emitted;
        for (const std::string &col : kOrder)
        {
            const bool present = (!csv_prop.empty() && csv_prop.count(col) != 0);
            if (present || csv_write_defaults)
            {
                columns.push_back(col);
                emitted.insert(col);
            }
        }
        if (!csv_prop.empty())
        {
            std::set<std::string> extra; // sorted -> deterministic
            for (const auto &kv : csv_prop)
                if (emitted.find(kv.first) == emitted.end())
                    extra.insert(kv.first);
            for (const std::string &col : extra)
                columns.push_back(col);
        }

        std::filesystem::path csv_path = obj_path;
        csv_path.replace_extension(".csv");

        std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
        if (!csv.is_open())
            throw std::runtime_error("Error opening file: failed to open '" + csv_path.string() + "'.");

        csv << "name";
        for (const std::string &col : columns)
            csv << "," << col;
        csv << "\n";

        for (arma::uword r = 0; r < n_csv; ++r)
        {
            csv << (csv_names)[r];
            for (const std::string &col : columns)
            {
                dtype val = default_for(col);
                if (!csv_prop.empty())
                    if (auto it = csv_prop.find(col); it != csv_prop.end())
                        val = it->second[r];
                csv << "," << fmt(val);
            }
            csv << "\n";
        }
        csv.close();
    }
}

template void quadriga_lib::obj_file_write(const std::string &fn,
                                           const arma::Mat<float> &mesh,
                                           const arma::uvec &obj_ind,
                                           const arma::uvec &mtl_ind,
                                           const std::vector<std::string> &obj_names,
                                           const std::vector<std::string> &mtl_names,
                                           arma::Mat<float> *vert_list_out,
                                           arma::umat *face_ind_out,
                                           const arma::Mat<float> &vert_list,
                                           const arma::umat &face_ind,
                                           const arma::Mat<float> &bsdf,
                                           double threshold,
                                           const arma::uvec &csv_ind,
                                           const std::vector<std::string> &csv_names,
                                           const std::unordered_map<std::string, std::vector<float>> &csv_prop,
                                           bool csv_write_defaults,
                                           bool split_loose_parts);

template void quadriga_lib::obj_file_write(const std::string &fn,
                                           const arma::Mat<double> &mesh,
                                           const arma::uvec &obj_ind,
                                           const arma::uvec &mtl_ind,
                                           const std::vector<std::string> &obj_names,
                                           const std::vector<std::string> &mtl_names,
                                           arma::Mat<double> *vert_list_out,
                                           arma::umat *face_ind_out,
                                           const arma::Mat<double> &vert_list,
                                           const arma::umat &face_ind,
                                           const arma::Mat<double> &bsdf,
                                           double threshold,
                                           const arma::uvec &csv_ind,
                                           const std::vector<std::string> &csv_names,
                                           const std::unordered_map<std::string, std::vector<double>> &csv_prop,
                                           bool csv_write_defaults,
                                           bool split_loose_parts);
