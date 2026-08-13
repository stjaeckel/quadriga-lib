// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <limits>
#include <cmath>       // std::floor
#include <cstdio>      // std::snprintf
#include <cstdlib>     // std::strtod
#include <type_traits> // std::is_same

// Helper: Convert mesh to vert_list + face_ind (Blender-style, per-object self-contained)
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
static void mesh2vert_list(const arma::Mat<dtype> &mesh, // mesh, Size: [ n_mesh, 9 ]
                           const arma::uvec &obj_ind,    // Object index, 0-based, Size: [ n_mesh ]
                           arma::Mat<dtype> &vert_list,  // Out: List of vertices, Size: [ n_vert_out, 3 ]
                           arma::umat &face_ind,         // Out: face indices, 0-based, Size: [ n_mesh, 3 ]
                           dtype threshold)              // Co-location threshold for vertices, Default: 1 mm
{
    const arma::uword n_mesh = mesh.n_rows;

    if (mesh.n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns.");

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
            return (size_t)(k.x * 73856093LL) ^ (size_t)(k.y * 19349663LL) ^ (size_t)(k.z * 83492791LL);
        }
    };

    face_ind.set_size(n_mesh, 3);

    // Per-object vertex coordinates, concatenated later
    std::vector<std::vector<dtype>> bvx(n_blk), bvy(n_blk), bvz(n_blk);

    // Weld each object independently; objects never share vertices
#pragma omp parallel for schedule(dynamic)
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

                const long long cx = (long long)std::floor((double)x * inv_cell);
                const long long cy = (long long)std::floor((double)y * inv_cell);
                const long long cz = (long long)std::floor((double)z * inv_cell);

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

    // Prefix offsets over the per-object vertex lists
    std::vector<arma::uword> offs(n_blk + 1, 0);
    for (size_t b = 0; b < n_blk; ++b)
        offs[b + 1] = offs[b] + (arma::uword)bvx[b].size();

    // Assemble output vertex list, Size: [ n_vert, 3 ]
    const arma::uword n_vert = offs[n_blk];
    vert_list.set_size(n_vert, 3);

#pragma omp parallel for schedule(dynamic)
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

// Helper: Split objects into connected components ("separate by loose parts")
// - Two faces belong to the same part if they share at least one vertex index; connectivity is
//  therefore defined by the welded vertex list, not by coordinates.
// - Parts never span objects, even if objects happen to share vertex indices.
// - Union-find over vertex indices, O(n_mesh * inverse-Ackermann); parts are numbered in order of
//  their first face, so the output order follows the input face order.
// - Parts are disjoint by construction, so splitting never duplicates vertices.
static void separate_loose_parts(const arma::umat &face_ind, // Face indices, Size: [ n_mesh, 3 ]
                                 const arma::uvec *obj_ind,  // Object index or nullptr, Size: [ n_mesh ]
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
        const PartKey key{find(face_ind.at(i, 0)), (obj_ind != nullptr) ? obj_ind->at(i) : (arma::uword)0};

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
  1-based material index (0 = no material, leaving that face unassigned); pass `mtl_ind = nullptr` to omit materials entirely
- The `.mtl` (named after the `.obj`) lists each used material; values default to a gray material when `bsdf` is omitted
- Duplicate entries in `mtl_names` are merged into one `.mtl` entry if their `bsdf` rows are identical (or if `bsdf`
  is omitted); duplicates with differing `bsdf` rows are disambiguated as `name.001`, `name.002`, ...
- If `csv_names` is given, the EM/acoustic material table is written to a companion `.csv` (named after the `.obj`):
  columns follow a fixed canonical order, then any extra `csv_prop` columns (alphabetical); `csv_write_defaults`
  additionally emits canonical columns absent from `csv_prop`, filled with their defaults (`a`, `e`, `fRef` = 1, else 0)

## Declaration:
```
void obj_file_write(
    const std::string &fn = "",
    const arma::Mat<dtype> *mesh = nullptr,
    const arma::uvec *obj_ind = nullptr,
    const arma::uvec *mtl_ind = nullptr,
    const std::vector<std::string> *obj_names = nullptr,
    const std::vector<std::string> *mtl_names = nullptr,
    arma::Mat<dtype> *vert_list_out = nullptr,
    arma::umat *face_ind_out = nullptr,
    const arma::Mat<dtype> *vert_list = nullptr,
    const arma::umat *face_ind = nullptr,
    const arma::Mat<dtype> *bsdf = nullptr,
    const dtype threshold = 0.001,
    const arma::uvec *csv_ind = nullptr,
    const std::vector<std::string> *csv_names = nullptr,
    const std::unordered_map<std::string, std::vector<dtype>> *csv_prop = nullptr,
    bool csv_write_defaults = false,
    bool split_loose_parts = false);
```

## Inputs:
- **`fn`** — Output path; must end in `.obj`; if empty, no files are written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{x1,y1,z1,...,x3,y3,z3}` per row; `[n_mesh, 9]`; mutually exclusive with `vert_list`/`face_ind`
- **`obj_ind`** — 0-based object index per face; `[n_mesh]`; each object must be a contiguous block
- **`mtl_ind`** — 1-based material index per face (0 = no material); `[n_mesh]`; omit (`nullptr`) for no materials
- **`obj_names`** — Object names; length > `max(obj_ind)`; required if `obj_ind` is given
- **`mtl_names`** — Material names; length ≥ `max(mtl_ind)` (1-based); required if `mtl_ind` is given
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
                                  const arma::Mat<dtype> *mesh,
                                  const arma::uvec *obj_ind,
                                  const arma::uvec *mtl_ind,
                                  const std::vector<std::string> *obj_names,
                                  const std::vector<std::string> *mtl_names,
                                  arma::Mat<dtype> *vert_list_out,
                                  arma::umat *face_ind_out,
                                  const arma::Mat<dtype> *vert_list,
                                  const arma::umat *face_ind,
                                  const arma::Mat<dtype> *bsdf,
                                  const dtype threshold,
                                  const arma::uvec *csv_ind,
                                  const std::vector<std::string> *csv_names,
                                  const std::unordered_map<std::string, std::vector<dtype>> *csv_prop,
                                  bool csv_write_defaults,
                                  bool split_loose_parts)
{
    // Mode selection: mesh XOR (vert_list + face_ind)
    const bool has_mesh = (mesh != nullptr);
    const bool has_vl = (vert_list != nullptr);
    const bool has_fi = (face_ind != nullptr);

    if (has_mesh && (has_vl || has_fi))
        throw std::invalid_argument("Provide either 'mesh' or 'vert_list'+'face_ind', not both.");
    if (!has_mesh && !has_vl && !has_fi)
        throw std::invalid_argument("Either 'mesh' or 'vert_list'+'face_ind' must be given.");
    if (!has_mesh && (!has_vl || !has_fi))
        throw std::invalid_argument("'vert_list' and 'face_ind' must be given together.");

    if (fn.empty() && vert_list_out == nullptr && face_ind_out == nullptr)
        return;

    // Number of faces + basic shape checks
    arma::uword n_mesh = 0;
    if (has_mesh)
    {
        if (mesh->n_cols != 9)
            throw std::invalid_argument("Input 'mesh' must have 9 columns.");
        n_mesh = mesh->n_rows;
    }
    else
    {
        if (vert_list->n_cols != 3)
            throw std::invalid_argument("Input 'vert_list' must have 3 columns.");
        if (face_ind->n_cols != 3)
            throw std::invalid_argument("Input 'face_ind' must have 3 columns.");
        n_mesh = face_ind->n_rows;
    }

    if (n_mesh == 0)
        throw std::invalid_argument("No faces to write (empty geometry).");

    // Validate obj_ind: 0-based, each object a contiguous block
    if (obj_ind != nullptr)
    {
        if (obj_ind->n_elem != n_mesh)
            throw std::invalid_argument("Input 'obj_ind' must have one element per face.");

        std::unordered_set<arma::uword> seen;
        arma::uword prev = obj_ind->at(0);
        seen.insert(prev);
        for (arma::uword n = 1; n < n_mesh; ++n)
        {
            const arma::uword cur = obj_ind->at(n);
            if (cur != prev)
            {
                if (!seen.insert(cur).second)
                    throw std::invalid_argument("Faces of each object must form a contiguous block in 'obj_ind'.");
                prev = cur;
            }
        }

        if (obj_names == nullptr || obj_names->size() <= obj_ind->max())
            throw std::invalid_argument("'obj_names' is missing or too short for the given 'obj_ind'.");
    }

    // Validate mtl_ind
    if (mtl_ind != nullptr)
    {
        if (mtl_ind->n_elem != n_mesh)
            throw std::invalid_argument("Input 'mtl_ind' must have one element per face.");

        const arma::uword n_mtl = mtl_ind->max(); // mtl_ind is 1-based; 0 = no material
        if (mtl_names == nullptr || mtl_names->size() < n_mtl)
            throw std::invalid_argument("'mtl_names' is missing or too short for the given 'mtl_ind'.");
        if (bsdf != nullptr && (bsdf->n_cols != 17 || bsdf->n_rows < n_mtl))
            throw std::invalid_argument("Input 'bsdf' must have 17 columns and one row per material.");
    }
    else if (bsdf != nullptr)
        throw std::invalid_argument("'bsdf' requires 'mtl_ind' and 'mtl_names'.");

    // Validate CSV material-table inputs (EM/acoustic side)
    const bool want_csv = (csv_ind != nullptr || csv_names != nullptr || csv_prop != nullptr);
    if (want_csv)
    {
        if (csv_names == nullptr || csv_names->empty())
            throw std::invalid_argument("Writing the material CSV requires a non-empty 'csv_names'.");

        const arma::uword n_csv = (arma::uword)csv_names->size();

        if (csv_prop != nullptr)
            for (const auto &kv : *csv_prop)
                if ((arma::uword)kv.second.size() != n_csv)
                    throw std::invalid_argument("Column '" + kv.first + "' in 'csv_prop' must have one value per entry in 'csv_names'.");

        if (csv_ind != nullptr)
        {
            if (csv_ind->n_elem != n_mesh)
                throw std::invalid_argument("Input 'csv_ind' must have one element per face.");
            if (csv_ind->max() > n_csv) // csv_ind is 1-based; 0 = no material
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
        arma::uvec obj_ones;
        const arma::uvec *pObj = obj_ind;
        if (pObj == nullptr)
        {
            obj_ones = arma::zeros<arma::uvec>(n_mesh);
            pObj = &obj_ones;
        }
        mesh2vert_list(*mesh, *pObj, VL_local, FI_local, threshold);
        pVL = &VL_local;
        pFI = &FI_local;
    }
    else
    {
        if (face_ind->max() >= vert_list->n_rows)
            throw std::invalid_argument("Input 'face_ind' references a vertex outside 'vert_list'.");
        pVL = vert_list;
        pFI = face_ind;
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
    { return (obj_ind != nullptr) ? obj_ind->at(f) : (arma::uword)0; };

    auto mtlid = [&](arma::uword f) -> arma::uword
    {
        if (mtl_ind == nullptr)
            return NO_MTL;
        const arma::uword v = mtl_ind->at(f);
        return (v == 0) ? NO_MTL : v - 1; // 1-based -> 0-based; 0 = no material
    };

    // Materials actually used (nonzero, sorted, unique)
    std::set<arma::uword> used_mtl;
    if (mtl_ind != nullptr)
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
            if (bsdf == nullptr)
                return true; // no BSDF data -> all same-named entries collapse into one
            for (arma::uword c = 0; c < 17; ++c)
                if (bsdf->at(p, c) != bsdf->at(q, c))
                    return false;
            return true;
        };

        // Group the used material rows by name (used_mtl is sorted, so rows stay ascending)
        std::map<std::string, std::vector<arma::uword>> by_name;
        for (const arma::uword id : used_mtl)
            by_name[(*mtl_names)[id]].push_back(id);

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
        throw std::invalid_argument("Error opening file: failed to open '" + fn + "'.");

    obj << "# Wavefront OBJ file written by quadriga-lib\n";
    if (write_materials)
        obj << "mtllib " << mtllib_name << "\n";

    const arma::Mat<dtype> &VL = *pVL;
    const arma::umat &FI = *pFI;

    // Blocks to write: "blk_bnd" delimits them in "face_order", "blk_obj" names their parent object.
    // Without splitting this is just the object blocks in input order (face_order = identity).
    std::vector<arma::uword> face_order, blk_bnd, blk_obj;

    if (split_loose_parts)
        separate_loose_parts(FI, obj_ind, VL.n_rows, face_order, blk_bnd, blk_obj);
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
        if (obj_ind != nullptr)
            return (*obj_names)[o];
        if (obj_names != nullptr && !obj_names->empty())
            return (*obj_names)[0];
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
        std::string last_mtl_name;
        bool have_last_mtl = false;
        for (arma::uword ii = bf; ii < bg; ++ii)
        {
            const arma::uword i = face_order[ii];

            if (write_materials)
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
            throw std::invalid_argument("Error opening file: failed to open '" + mtl_path.string() + "'.");

        mtl << "# Wavefront MTL file written by quadriga-lib\n";

        for (const arma::uword id : mtl_write_order)
        {
            mtl << "\nnewmtl " << mtl_out_name.at(id) << "\n";

            if (bsdf != nullptr)
            {
                const arma::uword r = id; // id is already the 0-based material row
                const dtype R = bsdf->at(r, 0), G = bsdf->at(r, 1), B = bsdf->at(r, 2);
                const dtype d = bsdf->at(r, 3), Pr = bsdf->at(r, 4), Pm = bsdf->at(r, 5);
                const dtype Ni = bsdf->at(r, 6), Ks = bsdf->at(r, 7);
                const dtype Re = bsdf->at(r, 8), Ge = bsdf->at(r, 9), Be = bsdf->at(r, 10);
                const dtype Ps = bsdf->at(r, 11), Pc = bsdf->at(r, 12), Pcr = bsdf->at(r, 13);
                const dtype an = bsdf->at(r, 14), anr = bsdf->at(r, 15), Tf = bsdf->at(r, 16);

                if (R != (dtype)0.8 || G != (dtype)0.8 || B != (dtype)0.8) // base color
                    mtl << "Kd " << fmt(R) << " " << fmt(G) << " " << fmt(B) << "\n";
                if (d != (dtype)1.0) // transparency
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
            "a", "b", "c", "d", "e", "f", "g", "h",
            "att", "attB", "alpha", "alphaB", "fRef", "m",
            "resF", "resQ", "resS", "coiF", "coiQ", "coiA", "tf", "tfB"};

        auto default_for = [](const std::string &col) -> dtype
        {
            if (col == "a" || col == "e" || col == "fRef")
                return (dtype)1.0;
            return (dtype)0.0;
        };

        const arma::uword n_csv = (arma::uword)csv_names->size();

        // Output columns: canonical order first (present, or all if csv_write_defaults), then extra csv_prop columns
        std::vector<std::string> columns;
        std::unordered_set<std::string> emitted;
        for (const std::string &col : kOrder)
        {
            const bool present = (csv_prop != nullptr && csv_prop->count(col) != 0);
            if (present || csv_write_defaults)
            {
                columns.push_back(col);
                emitted.insert(col);
            }
        }
        if (csv_prop != nullptr)
        {
            std::set<std::string> extra; // sorted -> deterministic
            for (const auto &kv : *csv_prop)
                if (emitted.find(kv.first) == emitted.end())
                    extra.insert(kv.first);
            for (const std::string &col : extra)
                columns.push_back(col);
        }

        std::filesystem::path csv_path = obj_path;
        csv_path.replace_extension(".csv");

        std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
        if (!csv.is_open())
            throw std::invalid_argument("Error opening file: failed to open '" + csv_path.string() + "'.");

        csv << "name";
        for (const std::string &col : columns)
            csv << "," << col;
        csv << "\n";

        for (arma::uword r = 0; r < n_csv; ++r)
        {
            csv << (*csv_names)[r];
            for (const std::string &col : columns)
            {
                dtype val = default_for(col);
                if (csv_prop != nullptr)
                    if (auto it = csv_prop->find(col); it != csv_prop->end())
                        val = it->second[r];
                csv << "," << fmt(val);
            }
            csv << "\n";
        }
        csv.close();
    }
}

template void quadriga_lib::obj_file_write(const std::string &fn,
                                           const arma::Mat<float> *mesh,
                                           const arma::uvec *obj_ind,
                                           const arma::uvec *mtl_ind,
                                           const std::vector<std::string> *obj_names,
                                           const std::vector<std::string> *mtl_names,
                                           arma::Mat<float> *vert_list_out,
                                           arma::umat *face_ind_out,
                                           const arma::Mat<float> *vert_list,
                                           const arma::umat *face_ind,
                                           const arma::Mat<float> *bsdf,
                                           const float threshold,
                                           const arma::uvec *csv_ind,
                                           const std::vector<std::string> *csv_names,
                                           const std::unordered_map<std::string, std::vector<float>> *csv_prop,
                                           bool csv_write_defaults,
                                           bool split_loose_parts);

template void quadriga_lib::obj_file_write(const std::string &fn,
                                           const arma::Mat<double> *mesh,
                                           const arma::uvec *obj_ind,
                                           const arma::uvec *mtl_ind,
                                           const std::vector<std::string> *obj_names,
                                           const std::vector<std::string> *mtl_names,
                                           arma::Mat<double> *vert_list_out,
                                           arma::umat *face_ind_out,
                                           const arma::Mat<double> *vert_list,
                                           const arma::umat *face_ind,
                                           const arma::Mat<double> *bsdf,
                                           const double threshold,
                                           const arma::uvec *csv_ind,
                                           const std::vector<std::string> *csv_names,
                                           const std::unordered_map<std::string, std::vector<double>> *csv_prop,
                                           bool csv_write_defaults,
                                           bool split_loose_parts);