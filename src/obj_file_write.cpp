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
#include <stdexcept>
#include <filesystem>
#include <charconv>

// Helper: Convert mesh to vert_list + face_ind (Blender-style, per-object self-contained)
// - Co-located vertices within "threshold" belonging to the SAME object are merged into one.
// - Identical coordinates in DIFFERENT objects are kept separate (duplicated); no cross-object
//  referencing, each object is self-contained.
// - Vertices are emitted object-by-object (in block order), so "vert_list" is grouped by object.
//  "face_ind" holds 0-based GLOBAL indices into "vert_list".
// - Requires the faces of each object to form a single contiguous block in "obj_ind"
//  (e.g. {1,1,2,2} is valid; {1,1,2,2,1} throws).
// - Greedy O(n^2) weld per object; not performance-critical.
template <typename dtype>
static void mesh2vert_list(const arma::Mat<dtype> &mesh, // mesh, Size: [ n_mesh, 9 ]
                           const arma::uvec &obj_ind,    // Object index, 1-based, Size: [ n_mesh ]
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

    // Global accumulated vertex coordinates
    std::vector<dtype> vx, vy, vz;
    vx.reserve(3 * n_mesh);
    vy.reserve(3 * n_mesh);
    vz.reserve(3 * n_mesh);

    face_ind.set_size(n_mesh, 3);

    // Global indices of vertices belonging to the current object (reset at each block)
    std::vector<arma::uword> obj_reps;
    arma::uword cur_obj = obj_ind.at(0);

    for (arma::uword n = 0; n < n_mesh; ++n)
    {
        if (obj_ind.at(n) != cur_obj) // entered a new object block
        {
            cur_obj = obj_ind.at(n);
            obj_reps.clear();
        }

        for (arma::uword k = 0; k < 3; ++k) // three triangle corners
        {
            const dtype x = mesh.at(n, 3 * k);
            const dtype y = mesh.at(n, 3 * k + 1);
            const dtype z = mesh.at(n, 3 * k + 2);

            // Search for a co-located vertex already added for this object
            bool found = false;
            arma::uword idx = 0;
            for (const arma::uword g : obj_reps)
            {
                const dtype dx = vx[g] - x, dy = vy[g] - y, dz = vz[g] - z;
                if (dx * dx + dy * dy + dz * dz <= threshold_sq)
                {
                    idx = g;
                    found = true;
                    break;
                }
            }

            if (!found) // add new vertex
            {
                idx = (arma::uword)vx.size();
                vx.push_back(x);
                vy.push_back(y);
                vz.push_back(z);
                obj_reps.push_back(idx);
            }

            face_ind.at(n, k) = idx; // 0-based global index
        }
    }

    // Assemble output vertex list, Size: [ n_vert, 3 ]
    const arma::uword n_vert = (arma::uword)vx.size();
    vert_list.set_size(n_vert, 3);
    for (arma::uword i = 0; i < n_vert; ++i)
    {
        vert_list.at(i, 0) = vx[i];
        vert_list.at(i, 1) = vy[i];
        vert_list.at(i, 2) = vz[i];
    }
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
- Without `mtl_ind` (or if all entries are `0`): no `usemtl` tags and no `.mtl` file are written;
  `mtl_ind = 0` marks an individual face as unassigned
- The `.mtl` (named after the `.obj`) lists each used material; values default to a gray material when `bsdf` is omitted

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
    const dtype threshold = 0.001);
```

## Inputs:
- **`fn`** — Output path; must end in `.obj`; if empty, no files are written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{x1,y1,z1,...,x3,y3,z3}` per row; `[n_mesh, 9]`; mutually exclusive with `vert_list`/`face_ind`
- **`obj_ind`** — 1-based object index per face; `[n_mesh]`; each object must be a contiguous block
- **`mtl_ind`** — 1-based material index per face (`0` = none); `[n_mesh]`
- **`obj_names`** — Object names; length ≥ `max(obj_ind)`; required if `obj_ind` is given
- **`mtl_names`** — Material names; length ≥ `max(mtl_ind)`; required if `mtl_ind` has nonzero entries
- **`vert_list`** — Vertex positions; `[n_vert, 3]`; only with `face_ind`, written unchanged
- **`face_ind`** — 0-based vertex indices per face; `[n_mesh, 3]`; required with `vert_list`
- **`bsdf`** — Principled BSDF for the `.mtl`; `[n_mtl, 17]`; see [[obj_file_read]] for columns
- **`threshold`** — Vertex co-location distance for merging within an object; default 1 mm

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
                                  const dtype threshold)
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

    // Validate obj_ind: 1-based, each object a contiguous block
    if (obj_ind != nullptr)
    {
        if (obj_ind->n_elem != n_mesh)
            throw std::invalid_argument("Input 'obj_ind' must have one element per face.");

        std::unordered_set<arma::uword> seen;
        arma::uword prev = obj_ind->at(0);
        if (prev == 0)
            throw std::invalid_argument("Input 'obj_ind' must be 1-based (found 0).");
        seen.insert(prev);
        for (arma::uword n = 1; n < n_mesh; ++n)
        {
            const arma::uword cur = obj_ind->at(n);
            if (cur == 0)
                throw std::invalid_argument("Input 'obj_ind' must be 1-based (found 0).");
            if (cur != prev)
            {
                if (!seen.insert(cur).second)
                    throw std::invalid_argument("Faces of each object must form a contiguous block in 'obj_ind'.");
                prev = cur;
            }
        }

        if (obj_names == nullptr || obj_names->size() < obj_ind->max())
            throw std::invalid_argument("'obj_names' is missing or too short for the given 'obj_ind'.");
    }

    // Validate mtl_ind (0 = no material, allowed)
    if (mtl_ind != nullptr)
    {
        if (mtl_ind->n_elem != n_mesh)
            throw std::invalid_argument("Input 'mtl_ind' must have one element per face.");

        const arma::uword max_mtl = mtl_ind->max();
        if (max_mtl > 0)
        {
            if (mtl_names == nullptr || mtl_names->size() < max_mtl)
                throw std::invalid_argument("'mtl_names' is missing or too short for the given 'mtl_ind'.");
            if (bsdf != nullptr && (bsdf->n_cols != 17 || bsdf->n_rows < max_mtl))
                throw std::invalid_argument("Input 'bsdf' must have 17 columns and one row per material.");
        }
    }
    else if (bsdf != nullptr)
        throw std::invalid_argument("'bsdf' requires 'mtl_ind' and 'mtl_names'.");

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
            obj_ones = arma::ones<arma::uvec>(n_mesh);
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
    auto objid = [&](arma::uword f) -> arma::uword
    { return (obj_ind != nullptr) ? obj_ind->at(f) : (arma::uword)1; };
    auto mtlid = [&](arma::uword f) -> arma::uword
    { return (mtl_ind != nullptr) ? mtl_ind->at(f) : (arma::uword)0; };

    // Materials actually used (nonzero, sorted, unique)
    std::set<arma::uword> used_mtl;
    if (mtl_ind != nullptr)
        for (arma::uword f = 0; f < n_mesh; ++f)
            if (const arma::uword m = mtlid(f); m != 0)
                used_mtl.insert(m);
    const bool write_materials = !used_mtl.empty();

    // Shortest round-trip number formatter (also maps -0 -> 0)
    auto fmt = [](dtype v) -> std::string
    {
        if (v == (dtype)0)
            return std::string("0");
        char buf[64];
        auto r = std::to_chars(buf, buf + sizeof(buf), v);
        return std::string(buf, r.ptr);
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

    arma::uword offset = 0; // cumulative vertices already written (global 1-based base)
    arma::uword f = 0;
    while (f < n_mesh)
    {
        const arma::uword cur_obj = objid(f);

        // Object block end (contiguous, guaranteed by the obj_ind guard)
        arma::uword g = f;
        while (g < n_mesh && objid(g) == cur_obj)
            ++g;

        // Object header
        std::string oname;
        if (obj_ind != nullptr)
            oname = (*obj_names)[cur_obj - 1];
        else if (obj_names != nullptr && !obj_names->empty())
            oname = (*obj_names)[0];
        else
            oname = "object";
        obj << "o " << oname << "\n";

        // Collect this object's vertices in first-use order
        std::unordered_map<arma::uword, arma::uword> remap;
        std::vector<arma::uword> order;
        order.reserve((g - f) * 3);
        for (arma::uword i = f; i < g; ++i)
            for (arma::uword k = 0; k < 3; ++k)
            {
                const arma::uword gv = FI.at(i, k);
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
        arma::uword last_mtl = 0;
        for (arma::uword i = f; i < g; ++i)
        {
            if (write_materials)
                if (const arma::uword m = mtlid(i); m != 0 && m != last_mtl)
                {
                    obj << "usemtl " << (*mtl_names)[m - 1] << "\n";
                    last_mtl = m;
                }

            const arma::uword a = offset + remap[FI.at(i, 0)] + 1;
            const arma::uword b = offset + remap[FI.at(i, 1)] + 1;
            const arma::uword c = offset + remap[FI.at(i, 2)] + 1;
            obj << "f " << a << " " << b << " " << c << "\n";
        }

        offset += (arma::uword)order.size();
        f = g;
    }
    obj.close();

    // Write .mtl (one newmtl per used material; defaults omitted)
    if (write_materials)
    {
        std::ofstream mtl(mtl_path, std::ios::out | std::ios::trunc);
        if (!mtl.is_open())
            throw std::invalid_argument("Error opening file: failed to open '" + mtl_path.string() + "'.");

        mtl << "# Wavefront MTL file written by quadriga-lib\n";

        for (const arma::uword id : used_mtl)
        {
            mtl << "\nnewmtl " << (*mtl_names)[id - 1] << "\n";

            if (bsdf != nullptr)
            {
                const arma::uword r = id - 1;
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
                                           const float threshold);

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
                                           const double threshold);