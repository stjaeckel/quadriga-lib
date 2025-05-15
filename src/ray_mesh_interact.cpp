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

#include <cstring> // For std::memcopy
#include <complex>
#include "quadriga_tools.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# ray_mesh_interact
Calculates interactions (reflection, transmission, refraction) of radio waves with objects

## Description:
- Radio waves that interact with a building, or other objects in the environment, will produce losses
  that depend on the electrical properties of the materials and material structure.
- This function calculates the interactions of radio-waves with objects in a 3D environment.
- It considers a plane wave incident upon a planar interface between two homogeneous and isotropic
  media of differing electric properties. The media extend sufficiently far from the interface such
  that the effect of any other interface is negligible.
- Supports beam-based modeling using triangular ray tubes defined by `trivec` and `tridir`.
- Air to media transition is assumed if front side of a face is hit and FBS != SBS
- Media to air transition is assumed if back side of a face is hit and FBS != SBS
- Media to media transition is assumed if FBS = SBS with opposing face orientations
- Order of the vertices determines side (front or back) of a mesh element
- Overlapping geometry in the triangle mesh must be avoided, since materials are transparent to radio
  waves.
- Implementation is done according to ITU-R P.2040-1.
- Rays that do not interact with the environment (i.e. for which `fbs_ind = 0`) are omitted from the output.
- Maintains consistency of input and output ray formats, including direction encoding (spherical or Cartesian).
- Assigns interaction type codes to output rays, allowing classification (e.g., single hit, total 
  reflection, material-to-material).
- Allowed datatypes (`dtype`): `float` or `double`.

## Declaration:
```
void quadriga_lib::ray_mesh_interact(
                int interaction_type,
                dtype center_frequency,
                const arma::Mat<dtype> *orig,
                const arma::Mat<dtype> *dest,
                const arma::Mat<dtype> *fbs,
                const arma::Mat<dtype> *sbs,
                const arma::Mat<dtype> *mesh,
                const arma::Mat<dtype> *mtl_prop,
                const arma::u32_vec *fbs_ind,
                const arma::u32_vec *sbs_ind,
                const arma::Mat<dtype> *trivec = nullptr,
                const arma::Mat<dtype> *tridir = nullptr,
                const arma::Col<dtype> *orig_length = nullptr,
                arma::Mat<dtype> *origN = nullptr,
                arma::Mat<dtype> *destN = nullptr,
                arma::Col<dtype> *gainN = nullptr,
                arma::Mat<dtype> *xprmatN = nullptr,
                arma::Mat<dtype> *trivecN = nullptr,
                arma::Mat<dtype> *tridirN = nullptr,
                arma::Col<dtype> *orig_lengthN = nullptr,
                arma::Col<dtype> *fbs_angleN = nullptr,
                arma::Col<dtype> *thicknessN = nullptr,
                arma::Col<dtype> *edge_lengthN = nullptr,
                arma::Mat<dtype> *normal_vecN = nullptr,
                arma::s32_vec *out_typeN = nullptr);
```

## Arguments:
- `int **interaction_type**` (input)<br>
  Type of interaction: 0 = reflection, 1 = transmission, 2 = refraction.

- `dtype **center_frequency**` (input)<br>
  Center frequency in Hz.

- `const arma::Mat<dtype> ***orig**, ***dest**` (input)<br>
  Ray origin and destination points in global Cartesian space. Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***fbs**, ***sbs**` (input)<br>
  First and second interaction points. Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***mesh**` (input)<br>
  Triangle mesh faces. Size: `[n_mesh, 9]`, See <a href="#obj_file_read">obj_file_read</a> for details

- `const arma::Mat<dtype> ***mtl_prop**` (input)<br>
  Material properties per face. Size: `[n_mesh, 5]`, See <a href="#obj_file_read">obj_file_read</a> for details

- `const arma::u32_vec ***fbs_ind**, ***sbs_ind**` (input)<br>
  Mesh indices of interaction points (1-based). Size: `[n_ray]`.

- `const arma::Mat<dtype> ***trivec** (optional input)<br>
  The 3 vectors pointing from the center point of the ray at the `origin` to the vertices of a triangular
  propagation tube (=beam wavefront), the values are in the order `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`;
  Size: `[no_ray, 9]`

- `const arma::Mat<dtype> ***tridir** (optional input)<br>
  The directions of the vertex-rays. Size: `[ n_ray, 6 ]` or `[ n_ray, 9 ]`,
  For 6 columns, values are in geographic coordinates (azimuth and elevation angle in rad); the 
  values are in the order `[ v1az, v1el, v2az, v2el, v3az, v3el ]`;
  For 9 columns, the input is in Cartesian coordinates and the values are in the  order 
  `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z  ]`

- `const arma::Col<dtype> ***orig_length**` (optional input)<br>
  Path length at origin. Size: `[n_ray]`. Default is 0

- `arma::Mat<dtype> ***origN**` (output)<br>
  New ray origins after the interaction with the medium, usually placed close to the FBS location.
  A small offset of 0.001 m in the direction of travel after the interaction with the medium is added
  to avoid getting stuct inside a mesh element. Size: `[ no_rayN, 3 ]`

- `arma::Mat<dtype> ***destN**` (output)<br>
  New ray destinaion after the interaction with the medium, taking the change of direction into account;
  Size: `[no_rayN, 3]`

- `arma::Col<dtype> ***gainN**` (output)<br>
  Gain (negative loss) caused by the interaction with the medium, averaged over both polarization
  directions. This value includes the in-medium attenuation, but does not account for FSPL. 
  Values are in linear scale (not dB); Size: `[no_rayN]`

- `arma::Mat<dtype> ***xprmatN**` (output)<br>
  Polarization transfer matrix; Interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH);
  The values account for the following effects: (1) gain caused by the interaction with the medium,
  (2) different reflection/transmission coefficients for transverse electric (TE) and transverse
  magnetic (TM) polarisation, (3) orientation of the incidence plane, (4) in-medium attenuation.
  FSPL is excluded, Size: `[ no_rayN, 8 ]`

- `arma::Mat<dtype> ***trivecN**, ***tridirN**` (output)<br>
  New ray tube geometry and direction. Format matches input.

- `arma::Col<dtype> ***orig_lengthN**` (output)<br>
  Length of the ray from `orig` to `origN`. If `orig_length` is given as input, its value is added.
  Size: `[ no_rayN ]`

- `arma::Col<dtype> ***fbs_angleN**` (output)<br>
  Angle between incoming ray and FBS in [rad], Size `[ n_rayN ]`

- `arma::Col<dtype> ***thicknessN**` (output)<br>
  Material thickness in meters calculated from the difference between FBS and SBS, Size `[ n_rayN ]`

- `arma::Col<dtype> ***edge_lengthN**` (output)<br>
  Max. edge length of the ray tube triangle at the new origin. A value of infinity indicates that only
  a part of the ray tube hits the object. Size `[ n_rayN, 3 ]`

- `arma::Mat<dtype> ***normal_vecN**` (output)<br>
  Normal vector of FBS and SBS, Size `[ n_rayN, 6 ]`

- `arma::s32_vec ***out_typeN**` (output)<br>
  A numeric indicator describing the type of the interaction. The total refection indicator is only 
  set in refraction mode. Size `[ n_rayN ]`<br><br>
  No | θF<0 | θS<0 | dFS=0 | TotRef | iSBS=0 | NF=-NS | NF=NS | startIn | endIn | Meaning
  ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
   0 |      |      |       |        |        |        |       |         |       | Undefined
   1 |   no |  N/A |    no |    N/A |    yes |    N/A |   N/A |      no |   yes | Single Hit o-i
   2 |  yes |  N/A |    no |     no |    yes |    N/A |   N/A |     yes |    no | Single Hit i-o
   3 |  yes |  N/A |    no |    yes |    yes |    N/A |   N/A |     yes |    no | Single Hit i-o, TR
  ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
   4 |   no |  yes |   yes |     no |     no |    yes |    no |     yes |   yes | M2M, M2 hit first
   5 |  yes |   no |   yes |     no |     no |    yes |    no |     yes |   yes | M2M, M1 hit first
   6 |  yes |   no |   yes |    yes |     no |    yes |    no |     yes |   yes | M2M, M1 hit first, TR
  ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
   7 |   no |   no |   yes |    N/A |     no |     no |   yes |      no |   yes | Overlapping Faces, o-i
   8 |  yes |  yes |   yes |     no |     no |     no |   yes |     yes |    no | Overlapping Faces, i-o
   9 |  yes |  yes |   yes |    yes |     no |     no |   yes |     yes |    no | Overlapping Faces, i-o, TR
  ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
  10 |   no |  yes |   yes |    N/A |     no |     no |    no |      no |    no | Edge Hit, o-i-o
  11 |  yes |   no |   yes |     no |     no |     no |    no |     yes |   yes | Edge Hit, i-o-i
  12 |  yes |   no |   yes |    yes |     no |     no |    no |     yes |   yes | Edge Hit, i-o-i, TR
  13 |   no |   no |   yes |    N/A |     no |     no |    no |      no |   yes | Edge Hit, o-i
  14 |  yes |  yes |   yes |     no |     no |     no |    no |     yes |    no | Edge Hit, i-o
  15 |  yes |  yes |   yes |    yes |     no |     no |    no |     yes |    no | Edge Hit, i-o, TR

## See also:
- <a href="#obj_file_read">obj_file_read</a> (for loading `mesh` and `mtl_prop` from OBJ file)
- <a href="#icosphere">icosphere</a> (for generating beams)
- <a href="#ray_triangle_intersect">ray_triangle_intersect</a> (for computing FBS and SBS positions)
- <a href="#ray_point_intersect">ray_point_intersect</a> (for calculating beam interactions with sampling points)
MD!*/

template <typename dtype>
void quadriga_lib::ray_mesh_interact(int interaction_type, dtype center_frequency,
                                     const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest, const arma::Mat<dtype> *fbs, const arma::Mat<dtype> *sbs,
                                     const arma::Mat<dtype> *mesh, const arma::Mat<dtype> *mtl_prop,
                                     const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                     const arma::Mat<dtype> *trivec, const arma::Mat<dtype> *tridir, const arma::Col<dtype> *orig_length,
                                     arma::Mat<dtype> *origN, arma::Mat<dtype> *destN, arma::Col<dtype> *gainN, arma::Mat<dtype> *xprmatN,
                                     arma::Mat<dtype> *trivecN, arma::Mat<dtype> *tridirN, arma::Col<dtype> *orig_lengthN,
                                     arma::Col<dtype> *fbs_angleN, arma::Col<dtype> *thicknessN, arma::Col<dtype> *edge_lengthN,
                                     arma::Mat<dtype> *normal_vecN, arma::s32_vec *out_typeN)
{
    // Ray offset is used to detect co-location of points, value in meters
    const double ray_offset = 0.001;

    // Check interaction_type
    if (interaction_type != 0 && interaction_type != 1 && interaction_type != 2)
        throw std::invalid_argument("Interaction type must be either (0) Reflection, (1) Transmission or (2) Refraction.");

    // Frequency in GHz
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    double fGHz = (double)center_frequency * 1.0e-9;

    // Check for NULL pointers
    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (dest == nullptr)
        throw std::invalid_argument("Input 'dest' cannot be NULL.");
    if (fbs == nullptr)
        throw std::invalid_argument("Input 'fbs' cannot be NULL.");
    if (sbs == nullptr)
        throw std::invalid_argument("Input 'sbs' cannot be NULL.");
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mtl_prop == nullptr)
        throw std::invalid_argument("Input 'mtl_prop' cannot be NULL.");
    if (fbs_ind == nullptr)
        throw std::invalid_argument("Input 'fbs_ind' cannot be NULL.");
    if (sbs_ind == nullptr)
        throw std::invalid_argument("Input 'sbs_ind' cannot be NULL.");

    // Check for correct number of columns
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    if (dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");
    if (fbs->n_cols != 3)
        throw std::invalid_argument("Input 'fbs' must have 3 columns containing x,y,z coordinates.");
    if (sbs->n_cols != 3)
        throw std::invalid_argument("Input 'sbs' must have 3 columns containing x,y,z coordinates.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mtl_prop->n_cols != 5)
        throw std::invalid_argument("Input 'mtl_prop' must have 5 columns.");

    const arma::uword n_ray = orig->n_rows;     // Number of rays
    const arma::uword n_mesh = mesh->n_rows;    // Number of mesh elements
    const int n_ray_i = (int)n_ray;             // Number of rays as int
    const unsigned n_mesh_u = (unsigned)n_mesh; // Number of mesh elements as unsigned int
    const size_t n_ray_t = (size_t)n_ray;       // Number of rays as size_t
    const size_t n_mesh_t = (size_t)n_mesh;     // Number of mesh elements as size_t

    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    // Check for correct number of rows
    if (dest->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");
    if (fbs->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'fbs' dont match.");
    if (sbs->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'sbs' dont match.");
    if (mtl_prop->n_rows != n_mesh)
        throw std::invalid_argument("Number of rows in 'mesh' and 'mtl_prop' dont match.");
    if (fbs_ind->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'fbs_ind' does not match number of rows in 'orig'.");
    if (sbs_ind->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'sbs_ind' does not match number of rows in 'orig'.");

    // Check input data for ray tube
    int use_ray_tube = 0;
    if (trivec != nullptr && !trivec->is_empty())
    {
        if (tridir == nullptr || tridir->is_empty())
            throw std::invalid_argument("In order to use ray tubes, both 'trivec' and 'tridir' must be given.");
        if (trivec->n_cols != 9)
            throw std::invalid_argument("Input 'trivec' must have 9 columns.");
        if (trivec->n_rows != n_ray)
            throw std::invalid_argument("Number of rows in 'orig' and 'trivec' dont match.");
        if (tridir->n_cols != 6 && tridir->n_cols != 9)
            throw std::invalid_argument("Input 'tridir' must have 6 or 9 columns.");
        if (tridir->n_rows != n_ray)
            throw std::invalid_argument("Number of rows in 'orig' and 'tridir' dont match.");
        use_ray_tube = (tridir->n_cols == 6) ? 1 : 2;
    }
    else if (tridir != nullptr && !tridir->is_empty())
        throw std::invalid_argument("In order to use ray tubes, both 'trivec' and 'tridir' must be given.");

    // Check for 'orig_length'
    if (orig_length != nullptr && !orig_length->is_empty() && orig_length->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'orig_length' does not match number of rows in 'orig'.");

    // Get input pointers
    const dtype *p_orig = orig->memptr();
    const dtype *p_dest = dest->memptr();
    const dtype *p_fbs = fbs->memptr();
    const dtype *p_sbs = sbs->memptr();
    const dtype *p_mesh = mesh->memptr();
    const dtype *p_mtl_prop = mtl_prop->memptr();
    const unsigned *p_fbs_ind = fbs_ind->memptr();
    const dtype *p_trivec = (trivec == nullptr) ? nullptr : trivec->memptr();
    const dtype *p_tridir = (tridir == nullptr) ? nullptr : tridir->memptr();
    const dtype *p_orig_length = (orig_length == nullptr) ? nullptr : orig_length->memptr();

    // Get number of output rays and build output ray index
    // - Only consider rays that have at least one interaction with the mesh, i.e. 'fbs_ind != 0'
    unsigned n_rayN_u = 0;
    unsigned *output_ray_index = new unsigned[n_ray_t]; // 1-based
    for (size_t i_ray = 0; i_ray < n_ray_t; ++i_ray)    // Ray loop
        if (p_fbs_ind[i_ray] == 0)                      // No hit
            output_ray_index[i_ray] = 0;
        else if (p_fbs_ind[i_ray] > n_mesh_u) // Invalid, must be 1 ... n_mesh (1-based index)
            throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");
        else // Store value
            output_ray_index[i_ray] = ++n_rayN_u;

    const arma::uword n_rayN = (arma::uword)n_rayN_u;
    const size_t n_rayN_t = (size_t)n_rayN_u;

    // Allocate output memory, if needed
    if (origN != nullptr && (origN->n_rows != n_rayN || origN->n_cols != 3))
        origN->set_size(n_rayN, 3);

    if (destN != nullptr && (destN->n_rows != n_rayN || destN->n_cols != 3))
        destN->set_size(n_rayN, 3);

    if (gainN != nullptr && gainN->n_elem != n_rayN)
        gainN->set_size(n_rayN);

    if (xprmatN != nullptr && (xprmatN->n_rows != n_rayN || xprmatN->n_cols != 8))
        xprmatN->set_size(n_rayN, 8);

    if (trivecN != nullptr && use_ray_tube && (trivecN->n_rows != n_rayN || trivecN->n_cols != 9))
        trivecN->set_size(n_rayN, 9);
    else if (trivecN != nullptr && !use_ray_tube && !trivecN->is_empty())
        trivecN->reset();

    if (tridirN != nullptr && use_ray_tube == 1 && (tridirN->n_rows != n_rayN || tridirN->n_cols != 6))
        tridirN->set_size(n_rayN, 6);
    else if (tridirN != nullptr && use_ray_tube == 2 && (tridirN->n_rows != n_rayN || tridirN->n_cols != 9))
        tridirN->set_size(n_rayN, 9);
    else if (tridirN != nullptr && !use_ray_tube && !tridirN->is_empty())
        tridirN->reset();

    if (orig_lengthN != nullptr && orig_lengthN->n_elem != n_rayN)
        orig_lengthN->set_size(n_rayN);

    if (fbs_angleN != nullptr && fbs_angleN->n_elem != n_rayN)
        fbs_angleN->set_size(n_rayN);

    if (thicknessN != nullptr && thicknessN->n_elem != n_rayN)
        thicknessN->set_size(n_rayN);

    if (edge_lengthN != nullptr && edge_lengthN->n_elem != n_rayN)
        edge_lengthN->set_size(n_rayN);

    if (normal_vecN != nullptr && (normal_vecN->n_rows != n_rayN || normal_vecN->n_cols != 6))
        normal_vecN->set_size(n_rayN, 6);

    if (out_typeN != nullptr && out_typeN->n_elem != n_rayN)
        out_typeN->set_size(n_rayN);

    // Get output pointers
    dtype *p_origN = (origN == nullptr) ? nullptr : origN->memptr();
    dtype *p_destN = (destN == nullptr) ? nullptr : destN->memptr();
    dtype *p_gainN = (gainN == nullptr) ? nullptr : gainN->memptr();
    dtype *p_xprmatN = (xprmatN == nullptr) ? nullptr : xprmatN->memptr();
    dtype *p_trivecN = (trivecN == nullptr) ? nullptr : trivecN->memptr();
    dtype *p_tridirN = (tridirN == nullptr) ? nullptr : tridirN->memptr();
    dtype *p_orig_lengthN = (orig_lengthN == nullptr) ? nullptr : orig_lengthN->memptr();
    dtype *p_fbs_angleN = (fbs_angleN == nullptr) ? nullptr : fbs_angleN->memptr();
    dtype *p_thicknessN = (thicknessN == nullptr) ? nullptr : thicknessN->memptr();
    dtype *p_edge_lengthN = (edge_lengthN == nullptr) ? nullptr : edge_lengthN->memptr();
    dtype *p_normal_vecN = (normal_vecN == nullptr) ? nullptr : normal_vecN->memptr();
    int *p_out_typeN = (out_typeN == nullptr) ? nullptr : out_typeN->memptr();

    // Only calculate ray tube if it is required in the output
    if (use_ray_tube && p_trivecN == nullptr && p_tridirN == nullptr)
        use_ray_tube = 0;

#pragma omp parallel for
    for (int i_ray = 0; i_ray < n_ray_i; ++i_ray) // Ray loop
    {
        if (p_fbs_ind[i_ray] == 0) // Skip non-hits
            continue;

        size_t iRx = (size_t)i_ray;                 // Ray x-index
        size_t iRy = iRx + n_ray_t;                 // Ray y-index
        size_t iRz = iRy + n_ray_t;                 // Ray z-index
        size_t iFBS = (size_t)p_fbs_ind[i_ray] - 1; // Mesh FBS index, 0-based

        // SBS index
        size_t iSBS = (size_t)sbs_ind->at(iRx); // Mesh SBS index, 1-based
        if (iSBS > n_mesh_t)
            throw std::invalid_argument("Some values in 'sbs_ind' exceed number of mesh elements.");

        double Ox = (double)p_orig[iRx], Oy = (double)p_orig[iRy], Oz = (double)p_orig[iRz]; // Origin position
        double Dx = (double)p_dest[iRx], Dy = (double)p_dest[iRy], Dz = (double)p_dest[iRz]; // Destination position
        double Fx = (double)p_fbs[iRx], Fy = (double)p_fbs[iRy], Fz = (double)p_fbs[iRz];    // FBS position
        double Sx = (double)p_sbs[iRx], Sy = (double)p_sbs[iRy], Sz = (double)p_sbs[iRz];    // SBS position
        double scl = 0.0;                                                                    // Scaling factor (reused)

        // Calculate normalized vector pointing from the origin to the FBS
        double OFx = Fx - Ox, OFy = Fy - Oy, OFz = Fz - Oz;              // Vector from origin to FBS (OF)
        double OF_length = std::sqrt(OFx * OFx + OFy * OFy + OFz * OFz); // Length of vector OF
        if (OF_length < ray_offset)                                      // Origin and FBS are co-located (rare case)
            OFx = Dx - Ox, OFy = Dy - Oy, OFz = Dz - Oz,                 // Assume that Destination is the FBS
                scl = 1.0 / std::sqrt(OFx * OFx + OFy * OFy + OFz * OFz);
        else
            scl = 1.0 / OF_length;
        OFx *= scl, OFy *= scl, OFz *= scl;

        // Calculate the length of the vector from FBS to SBS
        double FSx = Sx - Fx, FSy = Sy - Fy, FSz = Sz - Fz;              // Vector pointing from FBS to SBS
        double FS_length = std::sqrt(FSx * FSx + FSy * FSy + FSz * FSz); // Length of FS

        // Surface normal vector of the FBS mesh element calculated by taking the vector cross product of two edges of the triangle
        // Note: Order of the vertices determines side (front or back) of the element
        double V1x = (double)p_mesh[iFBS],
               V1y = (double)p_mesh[iFBS + n_mesh_t],
               V1z = (double)p_mesh[iFBS + 2 * n_mesh_t];
        double E1x = (double)p_mesh[iFBS + 3 * n_mesh_t] - V1x,
               E1y = (double)p_mesh[iFBS + 4 * n_mesh_t] - V1y,
               E1z = (double)p_mesh[iFBS + 5 * n_mesh_t] - V1z;
        double E2x = (double)p_mesh[iFBS + 6 * n_mesh_t] - V1x,
               E2y = (double)p_mesh[iFBS + 7 * n_mesh_t] - V1y,
               E2z = (double)p_mesh[iFBS + 8 * n_mesh_t] - V1z;
        double Nx = E1y * E2z - E1z * E2y, Ny = E1z * E2x - E1x * E2z, Nz = E1x * E2y - E1y * E2x; // Mesh surface normal
        scl = 1.0 / std::sqrt(Nx * Nx + Ny * Ny + Nz * Nz), Nx *= scl, Ny *= scl, Nz *= scl;       // Normalize to 1

        // Calculate incidence angle between surface of the mesh element at FBS and incoming ray
        double cos_theta = OFx * Nx + OFy * Ny + OFz * Nz;                           // Angle between normal vector and incoming ray
        cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta); // Boundary fix
        double theta = std::acos(cos_theta) - 1.570796326794897;                     // Angle between face and incoming ray, negative values illuminate back side

        // Calculate normal vector of the SBS mesh element, if needed
        double Mx = 0.0, My = 0.0, Mz = 0.0; // SBS normal vector
        double theta_sbs = 0.0;
        if (iSBS != 0 && (FS_length < ray_offset || p_normal_vecN != nullptr))
        {
            V1x = (double)p_mesh[iSBS - 1],
            V1y = (double)p_mesh[iSBS - 1 + n_mesh_t],
            V1z = (double)p_mesh[iSBS - 1 + 2 * n_mesh_t];
            E1x = (double)p_mesh[iSBS - 1 + 3 * n_mesh_t] - V1x,
            E1y = (double)p_mesh[iSBS - 1 + 4 * n_mesh_t] - V1y,
            E1z = (double)p_mesh[iSBS - 1 + 5 * n_mesh_t] - V1z;
            E2x = (double)p_mesh[iSBS - 1 + 6 * n_mesh_t] - V1x,
            E2y = (double)p_mesh[iSBS - 1 + 7 * n_mesh_t] - V1y,
            E2z = (double)p_mesh[iSBS - 1 + 8 * n_mesh_t] - V1z;
            Mx = E1y * E2z - E1z * E2y, My = E1z * E2x - E1x * E2z, Mz = E1x * E2y - E1y * E2x;  // Mesh surface normal
            scl = 1.0 / std::sqrt(Mx * Mx + My * My + Mz * Mz), Mx *= scl, My *= scl, Mz *= scl; // Normalize to 1

            // Incidence angle at SBS
            theta_sbs = OFx * Mx + OFy * My + OFz * Mz;
            theta_sbs = (theta_sbs < -1.0) ? -1.0 : (theta_sbs > 1.0 ? 1.0 : theta_sbs);
            theta_sbs = std::acos(theta_sbs) - 1.570796326794897;
        }

        // Determine the type of the interaction
        int out_type = (theta >= 0.0) ? 1 : (theta < 0.0 ? 2 : 0); // Output type (0 = undefined, 1 = outside to inside, 2 = inside to outside)
        bool material_to_material = false;                         // Assume no material to material transition
        bool ray_starts_inside = theta < 0.0;                      // Hitting a face back side at the FBS is a certain sign
        if (FS_length < ray_offset && iSBS != 0)                   // Two colocated faces
        {
            const double lim = 1.0e-4;
            if (std::abs(Nx + Mx) < lim && std::abs(Ny + My) < lim && std::abs(Nz + Mz) < lim) // Opposing normal vectors = material to material transition
                material_to_material = true, ray_starts_inside = true,
                out_type = (theta >= 0.0) ? 4 : (theta < 0.0 ? 5 : 0);
            else if (std::abs(Nx - Mx) < lim && std::abs(Ny - My) < lim && std::abs(Nz - Mz) < lim) // Equal normal vectors = overlapping or duplicate faces
                out_type = (theta >= 0.0) ? 7 : (theta < 0.0 ? 8 : 0);
            else if (theta >= 0.0 && theta_sbs <= 0.0)
                out_type = 10; // Edge Hit, o-i-o
            else if (theta < 0.0 && theta_sbs >= 0.0)
                out_type = 11; // Edge Hit, i-o-i
            else if ((theta >= 0.0 && theta_sbs >= 0.0))
                out_type = 13; // Edge Hit, o-i
            else if ((theta < 0.0 && theta_sbs <= 0.0))
                out_type = 14; // Edge Hit, i-o
            else               // Undefined state
                out_type = 0;
        }

        // Flip normal vector in case of back side illumination
        if (theta < 0.0)
            Nx = -Nx, Ny = -Ny, Nz = -Nz,
            cos_theta = OFx * Nx + OFy * Ny + OFz * Nz,
            cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta);

        // Limit value to 0 ... 1 for calculating reflection and transmission coefficients
        double abs_cos_theta = std::abs(cos_theta);

        // Select the properties of the two materials
        double kR1 = 1.0, kR2 = 0.0, kR3 = 0.0, kR4 = 0.0; // First material properties : air
        double kS1 = 1.0, kS2 = 0.0, kS3 = 0.0, kS4 = 0.0; // Second material properties : air
        double transition_gain = 1.0;                      // Additional gain for face transition, linear scale

        if (theta >= 0.0) // Ray hits front side of FBS/SBS face, set second material to object material
        {
            kS1 = (double)p_mtl_prop[iFBS];
            kS2 = (double)p_mtl_prop[iFBS + n_mesh_t];
            kS3 = (double)p_mtl_prop[iFBS + 2 * n_mesh_t];
            kS4 = (double)p_mtl_prop[iFBS + 3 * n_mesh_t];
            transition_gain = std::pow(10.0, -0.1 * (double)p_mtl_prop[iFBS + 4 * n_mesh_t]);
        }
        else // Ray hits back side of FBS face, set first material to object material
        {
            kR1 = (double)p_mtl_prop[iFBS];
            kR2 = (double)p_mtl_prop[iFBS + n_mesh_t];
            kR3 = (double)p_mtl_prop[iFBS + 2 * n_mesh_t];
            kR4 = (double)p_mtl_prop[iFBS + 3 * n_mesh_t];
        }

        if (material_to_material) // Material to material transition
        {
            if (theta >= 0.0) // SBS (front side) is hit first
            {
                kR1 = (double)p_mtl_prop[iSBS - 1];
                kR2 = (double)p_mtl_prop[iSBS - 1 + n_mesh_t];
                kR3 = (double)p_mtl_prop[iSBS - 1 + 2 * n_mesh_t];
                kR4 = (double)p_mtl_prop[iSBS - 1 + 3 * n_mesh_t];
            }
            else // FBS (back side) is hit first
            {
                kS1 = (double)p_mtl_prop[iSBS - 1];
                kS2 = (double)p_mtl_prop[iSBS - 1 + n_mesh_t];
                kS3 = (double)p_mtl_prop[iSBS - 1 + 2 * n_mesh_t];
                kS4 = (double)p_mtl_prop[iSBS - 1 + 3 * n_mesh_t];
                transition_gain = std::pow(10.0, -0.1 * (double)p_mtl_prop[iSBS - 1 + 4 * n_mesh_t]);
            }
        }

        // Calculate complex-valued relative permittivity of medium 1 and 2, ITU-R P.2040-1, eq. (9b)
        scl = -17.98 / fGHz;
        std::complex<double> eta1(kR1 * std::pow(fGHz, kR2), scl * kR3 * std::pow(fGHz, kR4)); // Material 1
        std::complex<double> eta2(kS1 * std::pow(fGHz, kS2), scl * kS3 * std::pow(fGHz, kS4)); // Material 2
        bool dense_to_light = std::real(eta1) > std::real(eta2);

        // Evaluate total reflection condition in ITU-R P.2040-1, eq. (31) and (32)
        double sin_theta = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta); // Trigonometric identity
        std::complex<double> eta1_div_eta2 = eta1 / eta2;                  // Complex division
        double eta = std::sqrt(std::abs(eta1_div_eta2));                   // sgrt( abs( eta1 / eta2 ) )
        bool total_reflection = eta * sin_theta >= 1.0;                    // Total reflection condition

        // Calculate cos_theta2 from Rec. ITU-R P.2040-1, eq. (33)
        std::complex<double> cos_theta2 = std::sqrt(1.0 - eta1_div_eta2 * sin_theta * sin_theta);

        // Calculate the center path direction after medium interaction (normalized to length 1)
        double FDx = Dx - Fx, FDy = Dy - Fy, FDz = Dz - Fz;              // Vector from FBS to destination
        double FD_length = std::sqrt(FDx * FDx + FDy * FDy + FDz * FDz); // Length of path from FBS to destination

        if (interaction_type == 0) // Reflection, normalized by default
            FDx = OFx - 2.0 * cos_theta * Nx,
            FDy = OFy - 2.0 * cos_theta * Ny,
            FDz = OFz - 2.0 * cos_theta * Nz;
        else if (interaction_type == 1)      // Transmission without refraction
            FDx = OFx, FDy = OFy, FDz = OFz; // New path direction = same as incoming ray, already normalized
        else                                 // Refraction
        {
            scl = eta * abs_cos_theta - std::real(cos_theta2);                                            // Temporary variable, ignoring imaginary part of cos_theta2
            FDx = eta * OFx + scl * Nx, FDy = eta * OFy + scl * Ny, FDz = eta * OFz + scl * Nz;           // Refraction into medium
            scl = 1.0 / std::sqrt(FDx * FDx + FDy * FDy + FDz * FDz), FDx *= scl, FDy *= scl, FDz *= scl; // Normalize
        }

        // Update origin and direction of the ray tube vertices
        double p_trivec_tmp[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double p_tridir_tmp[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double edge_length_tmp = 0.0;
        if (use_ray_tube)
        {
            // Process each vertex-ray separately
            for (int iTube = 1; iTube <= 3; ++iTube)
            {
                // Load origin and direction
                double Tx = Ox, Ty = Oy, Tz = Oz, az = 0.0, el = 0.0, Vx = 0.0, Vy = 0.0, Vz = 0.0;
                if (iTube == 1)
                {
                    Tx += (double)p_trivec[iRx], Ty += (double)p_trivec[iRy], Tz += (double)p_trivec[iRz];
                    if (use_ray_tube == 1)
                        az = (double)p_tridir[iRx], el = (double)p_tridir[iRy];
                    else
                        Vx = (double)p_tridir[iRx], Vy = (double)p_tridir[iRy], Vz = (double)p_tridir[iRz];
                }
                else if (iTube == 2)
                {
                    Tx += (double)p_trivec[iRx + 3 * n_ray_t], Ty += (double)p_trivec[iRx + 4 * n_ray_t], Tz += (double)p_trivec[iRx + 5 * n_ray_t];
                    if (use_ray_tube == 1)
                        az = (double)p_tridir[iRx + 2 * n_ray_t], el = (double)p_tridir[iRx + 3 * n_ray_t];
                    else
                        Vx = (double)p_tridir[iRx + 3 * n_ray_t], Vy = (double)p_tridir[iRx + 4 * n_ray_t], Vz = (double)p_tridir[iRx + 5 * n_ray_t];
                }
                else if (iTube == 3)
                {
                    Tx += (double)p_trivec[iRx + 6 * n_ray_t], Ty += (double)p_trivec[iRx + 7 * n_ray_t], Tz += (double)p_trivec[iRx + 8 * n_ray_t];
                    if (use_ray_tube == 1)
                        az = (double)p_tridir[iRx + 4 * n_ray_t], el = (double)p_tridir[iRx + 5 * n_ray_t];
                    else
                        Vx = (double)p_tridir[iRx + 6 * n_ray_t], Vy = (double)p_tridir[iRx + 7 * n_ray_t], Vz = (double)p_tridir[iRx + 8 * n_ray_t];
                }

                // Calculate vertex ray direction (V)
                if (use_ray_tube == 1) // Spherical input
                {
                    scl = std::cos(el);
                    Vx = std::cos(az) * scl, Vy = std::sin(az) * scl, Vz = std::sin(el);
                }
                else // Cartesian input
                {
                    double scl = Vx * Vx + Vy * Vy + Vz * Vz;
                    if (std::abs(scl - 1.0) > 2e-7)
                    {
                        scl = 1.0 / std::sqrt(scl);
                        Vx *= scl, Vy *= scl, Vz *= scl; // Normalize
                    }
                }

                // Calculate intersect point of the vertex-ray with the face
                double d = ((Fx - Tx) * Nx + (Fy - Ty) * Ny + (Fz - Tz) * Nz) / (Vx * Nx + Vy * Ny + Vz * Nz); // Distance from vert. origin to face (d)
                double Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;                                   // Intersect point with face (W)

                if (d < 0.0 || d > 1.0e5) // Vertex ray does not hit face
                    edge_length_tmp = INFINITY;

                if (interaction_type == 0) // Reflection
                {
                    if (d < 0.0 || d > 1.0e5) // Ray does not hit face - use orthogonal projection on ray
                    {
                        d = ((Fx - Tx) * Vx + (Fy - Ty) * Vy + (Fz - Tz) * Vz) / (Vx * Vx + Vy * Vy + Vz * Vz);
                        Tx = Tx + Vx * d - Fx, Ty = Ty + Vy * d - Fy, Tz = Tz + Vz * d - Fz; // Scaled vertex - updates T
                        double a = 2.0 * (Tx * Nx + Ty * Ny + Tz * Nz);                      // Reflection of T on face
                        Tx -= a * Nx + ray_offset * FDx;
                        Ty -= a * Ny + ray_offset * FDy;
                        Tz -= a * Nz + ray_offset * FDz;
                    }
                    else // Use intersect point W as new vertex origin
                    {
                        Tx = Wx - Fx - ray_offset * FDx;
                        Ty = Wy - Fy - ray_offset * FDy;
                        Tz = Wz - Fz - ray_offset * FDz;
                    }

                    // Update vertex direction
                    double a = 2.0 * (Vx * Nx + Vy * Ny + Vz * Nz);
                    Vx -= a * Nx, Vy -= a * Ny, Vz -= a * Nz;
                    if (use_ray_tube == 1)
                    {
                        Vz = (Vz < -1.0) ? -1.0 : (Vz > 1.0 ? 1.0 : Vz); // Boundary fix
                        az = std::atan2(Vy, Vx), el = std::asin(Vz);
                    }
                }
                else // Transmission and Refraction
                {
                    if (d < 0.0 || d > 1.0e5) // Ray does not hit face - use orthogonal projection on vertex ray
                    {
                        d = ((Fx - Tx) * Vx + (Fy - Ty) * Vy + (Fz - Tz) * Vz) / (Vx * Vx + Vy * Vy + Vz * Vz);
                        Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
                    }

                    // Update ray tube coordinates
                    Tx = Wx - Fx - ray_offset * FDx;
                    Ty = Wy - Fy - ray_offset * FDy;
                    Tz = Wz - Fz - ray_offset * FDz;

                    // Vertex ray directions remains the same for Transmission
                    if (interaction_type == 2) // Refraction
                    {
                        double cos_thetaV = std::abs(Vx * Nx + Vy * Ny + Vz * Nz);       // Cosine of incidence angle
                        double sin_thetaV = std::sqrt(1.0 - cos_thetaV * cos_thetaV);    // Sine of incidence angle
                        total_reflection = total_reflection | (eta * sin_thetaV >= 1.0); // Check total reflection condition

                        // Refraction into medium
                        std::complex<double> cos_theta2V = std::sqrt(1.0 - eta1_div_eta2 * sin_thetaV * sin_thetaV);
                        double scl = eta * cos_thetaV - std::real(cos_theta2V);
                        Vx = eta * Vx + scl * Nx, Vy = eta * Vy + scl * Ny, Vz = eta * Vz + scl * Nz;

                        scl = 1.0 / std::sqrt(Vx * Vx + Vy * Vy + Vz * Vz);
                        Vx *= scl, Vy *= scl, Vz *= scl; // Normalize
                        if (use_ray_tube == 1)
                        {
                            Vz = (Vz < -1.0) ? -1.0 : (Vz > 1.0 ? 1.0 : Vz); // Boundary fix
                            az = std::atan2(Vy, Vx), el = std::asin(Vz);     // Angles
                        }
                    }
                }

                // Write new vertex ray origin and direction - convert back to dtype
                if (iTube == 1)
                {
                    p_trivec_tmp[0] = Tx, p_trivec_tmp[1] = Ty, p_trivec_tmp[2] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[0] = az, p_tridir_tmp[1] = el;
                    else
                        p_tridir_tmp[0] = Vx, p_tridir_tmp[1] = Vy, p_tridir_tmp[2] = Vz;
                }
                else if (iTube == 2)
                {
                    p_trivec_tmp[3] = Tx, p_trivec_tmp[4] = Ty, p_trivec_tmp[5] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[2] = az, p_tridir_tmp[3] = el;
                    else
                        p_tridir_tmp[3] = Vx, p_tridir_tmp[4] = Vy, p_tridir_tmp[5] = Vz;
                }
                else if (iTube == 3)
                {
                    p_trivec_tmp[6] = Tx, p_trivec_tmp[7] = Ty, p_trivec_tmp[8] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[4] = az, p_tridir_tmp[5] = el;
                    else
                        p_tridir_tmp[6] = Vx, p_tridir_tmp[7] = Vy, p_tridir_tmp[8] = Vz;
                }
            }

            // Calculate the maximum edge length
            if (p_edge_lengthN != nullptr)
            {
                double Ex = p_trivec_tmp[3] - p_trivec_tmp[0], Ey = p_trivec_tmp[4] - p_trivec_tmp[1], Ez = p_trivec_tmp[5] - p_trivec_tmp[2];
                scl = Ex * Ex + Ey * Ey + Ez * Ez;
                edge_length_tmp = (scl > edge_length_tmp) ? scl : edge_length_tmp;
                Ex = p_trivec_tmp[6] - p_trivec_tmp[0], Ey = p_trivec_tmp[7] - p_trivec_tmp[1], Ez = p_trivec_tmp[8] - p_trivec_tmp[2];
                scl = Ex * Ex + Ey * Ey + Ez * Ez;
                edge_length_tmp = (scl > edge_length_tmp) ? scl : edge_length_tmp;
                Ex = p_trivec_tmp[6] - p_trivec_tmp[3], Ey = p_trivec_tmp[7] - p_trivec_tmp[4], Ez = p_trivec_tmp[8] - p_trivec_tmp[5];
                scl = Ex * Ex + Ey * Ey + Ez * Ez;
                edge_length_tmp = (scl > edge_length_tmp) ? scl : edge_length_tmp;
                edge_length_tmp = std::sqrt(edge_length_tmp);
            }
        }

        // Determine the in-medium gain
        double gain = 1.0;     // Initial gain
        if (ray_starts_inside) // First medium attenuation, from origin to FBS
        {
            // First medium attenuation, from origin to FBS
            // in case of reflection, ray continues in the same medium (account for ray offset)
            double thickness = (interaction_type == 0) ? OF_length + ray_offset : OF_length;
            scl = std::real(eta1);
            double tan_delta = std::imag(eta1) / scl;                        // Loss tangent
            double cos_delta = 1.0 / std::sqrt(1.0 + tan_delta * tan_delta); // Trigonometric identity
            double Delta = 2.0 * cos_delta / (1.0 - cos_delta);              // Attenuation distance (1)
            Delta = std::sqrt(Delta) * 0.0477135 / (fGHz * std::sqrt(scl));  // Attenuation distance (2)
            gain *= std::pow(10.0, -0.1 * thickness * 8.686 / Delta);        // Gain update
        }
        if (interaction_type != 0) // Attenuation caused by the medium after transition
        {
            scl = std::real(eta2);
            double tan_delta = std::imag(eta2) / scl;                        // Loss tangent
            double cos_delta = 1.0 / std::sqrt(1.0 + tan_delta * tan_delta); // Trigonometric identity
            double Delta = 2.0 * cos_delta / (1.0 - cos_delta);              // Attenuation distance (1)
            Delta = std::sqrt(Delta) * 0.0477135 / (fGHz * std::sqrt(scl));  // Attenuation distance (2)
            gain *= std::pow(10.0, -0.1 * ray_offset * 8.686 / Delta);       // Gain update
        }

        // Add additional transition gain
        if (interaction_type != 0) // Only for transmission and refraction
            gain *= transition_gain;

        // Calculate sqrt(eta1) and sqrt(eta2) needed for ITU-R P.2040-1, eq. (31) and (32)
        eta1 = std::sqrt(eta1); // Complex square root
        eta2 = std::sqrt(eta2); // Complex square root

        // Calculate Reflection coefficients  ITU-R P.2040-1, eq. (31)
        std::complex<double> R_eTE = total_reflection ? 1.0 : 0.0;
        std::complex<double> R_eTM = total_reflection ? 1.0 : 0.0;
        double reflection_gain = total_reflection ? 1.0 : 0.0;
        if (interaction_type == 1 || (interaction_type == 0 && !total_reflection)) // Reflection and Transmission
            R_eTE = (eta1 * abs_cos_theta - eta2 * cos_theta2) / (eta1 * abs_cos_theta + eta2 * cos_theta2),
            R_eTM = (eta2 * abs_cos_theta - eta1 * cos_theta2) / (eta2 * abs_cos_theta + eta1 * cos_theta2),
            reflection_gain = 0.5 * (std::norm(R_eTE) + std::norm(R_eTM));

        // Calculate Transmission coefficients  ITU-R P.2040-1, eq. (32)
        std::complex<double> T_eTE(0.0, 0.0), T_eTM(0.0, 0.0);
        double refraction_gain = 0.0;
        if (!total_reflection && interaction_type != 0) // Transmission and Refraction
            T_eTE = (2.0 * eta1 * abs_cos_theta) / (eta1 * abs_cos_theta + eta2 * cos_theta2),
            T_eTM = (2.0 * eta1 * abs_cos_theta) / (eta2 * abs_cos_theta + eta1 * cos_theta2),
            refraction_gain = 0.5 * (std::norm(T_eTE) + std::norm(T_eTM));

        // Special Case: Transmission from a dense medium (such as glass) to a light medium (e.g. air)
        // Transmission mode assumed no change of direction (unlike refraction) and no total reflection is possible.
        // This may violate the conservation of energy principle (e.g. cause false amplification).
        // To obtain meaningful results, we ignore the losses in this case.
        if (interaction_type == 1 && dense_to_light)
            T_eTE = 1.0, T_eTM = 1.0, refraction_gain = 1.0, reflection_gain = 0.0;

        // Select corresponding type
        double eTE_Re = (interaction_type == 0) ? std::real(R_eTE) : std::real(T_eTE),
               eTE_Im = (interaction_type == 0) ? std::imag(R_eTE) : std::imag(T_eTE),
               eTM_Re = (interaction_type == 0) ? std::real(R_eTM) : std::real(T_eTM),
               eTM_Im = (interaction_type == 0) ? std::imag(R_eTM) : std::imag(T_eTM);

        // Read the output ray index
        size_t i_rayN = output_ray_index[iRx] - 1; // Output ray index, 0-based
        if (i_rayN >= n_rayN_t)                    // Just to be sure to avoid any segfaults
            throw std::invalid_argument("Something went wrong. This should never be reached!");

        // Write origN, add a small offset to prevent it from getting stuck inside the mesh element
        if (p_origN != nullptr)
        {
            p_origN[i_rayN] = dtype(Fx + ray_offset * FDx);
            p_origN[i_rayN + n_rayN_t] = dtype(Fy + ray_offset * FDy);
            p_origN[i_rayN + 2 * n_rayN_t] = dtype(Fz + ray_offset * FDz);
        }

        // Write destN
        if (p_destN != nullptr)
        {
            // Make sure the new destination is beyond the new start point
            FD_length = (FD_length <= ray_offset) ? 2.0 * ray_offset : FD_length;
            p_destN[i_rayN] = dtype(Fx + FD_length * FDx);
            p_destN[i_rayN + n_rayN_t] = dtype(Fy + FD_length * FDy);
            p_destN[i_rayN + 2 * n_rayN_t] = dtype(Fz + FD_length * FDz);
        }

        if (p_gainN != nullptr)
        {
            // Include average medium transition gain
            if (interaction_type == 0)
                scl = gain * reflection_gain;
            else if (interaction_type == 1)
                scl = gain * (1.0 - reflection_gain);
            else
                scl = gain * refraction_gain;

            p_gainN[i_rayN] = dtype(scl);
        }

        if (p_xprmatN != nullptr)
        {
            // Calculate vectors for the polarization base transformation (incoming path)
            double Hx = -OFy + 3.0e-20, Hy = OFx, Hz = 0.0;                                                // Polarization base in ePhi direction (horizontal)
            scl = 1.0 / std::sqrt(Hx * Hx + Hy * Hy), Hx *= scl, Hy *= scl;                                // Normalize
            double Vx = -OFz * Hy, Vy = OFz * Hx, Vz = OFx * Hy - OFy * Hx;                                // Polarization base in eTheta direction (vertical)
            double Qx = OFy * Nz - OFz * Ny + 3.0e-20, Qy = OFz * Nx - OFx * Nz, Qz = OFx * Ny - OFy * Nx; // Base vector perpendicular to plane normal (eQ)
            scl = 1.0 / std::sqrt(Qx * Qx + Qy * Qy + Qz * Qz), Qx *= scl, Qy *= scl, Qz *= scl;           // Normalize
            double Px = Qy * OFz - Qz * OFy, Py = Qz * OFx - Qx * OFz, Pz = Qx * OFy - Qy * OFx;           // Base vector parallel to plane normal (eP)

            // Calculate polarization base transformation matrix from global coordinates to local coordinates
            bool do_base_transform = scl < 1.0e19;
            double Q1 = (do_base_transform) ? Vx * Px + Vy * Py + Vz * Pz : 1.0; // dot( eV, eP )
            double Q2 = (do_base_transform) ? Vx * Qx + Vy * Qy + Vz * Qz : 0.0; // dot( eV, eQ )
            double Q3 = (do_base_transform) ? Hx * Px + Hy * Py + Hz * Pz : 0.0; // dot( eH, eP )
            double Q4 = (do_base_transform) ? Hx * Qx + Hy * Qy + Hz * Qz : 1.0; // dot( eH, eQ )

            // Calculate vectors for the polarization base transformation (outgoing path)
            Hx = -FDy + 3.0e-20, Hy = FDx, Hz = 0.0;                                                // Polarization base in ePhi direction (horizontal)
            scl = 1.0 / std::sqrt(Hx * Hx + Hy * Hy), Hx *= scl, Hy *= scl;                         // Normalize
            Vx = -FDz * Hy, Vy = FDz * Hx, Vz = FDx * Hy - FDy * Hx;                                // Polarization base in eTheta direction (vertical)
            Qx = FDy * Nz - FDz * Ny + 3.0e-20, Qy = FDz * Nx - FDx * Nz, Qz = FDx * Ny - FDy * Nx; // Base vector perpendicular to plane normal (eQ)
            scl = 1.0 / std::sqrt(Qx * Qx + Qy * Qy + Qz * Qz), Qx *= scl, Qy *= scl, Qz *= scl;    // Normalize
            Px = Qy * FDz - Qz * FDy, Py = Qz * FDx - Qx * FDz, Pz = Qx * FDy - Qy * FDx;           // Base vector parallel to plane normal (eP)

            // Calculate polarization base transformation matrix from global coordinates to local coordinates
            do_base_transform = scl < 1.0e19;
            double U1 = (do_base_transform) ? Vx * Px + Vy * Py + Vz * Pz : 1.0; // dot( eV, eP )
            double U2 = (do_base_transform) ? Vx * Qx + Vy * Qy + Vz * Qz : 0.0; // dot( eV, eQ )
            double U3 = (do_base_transform) ? Hx * Px + Hy * Py + Hz * Pz : 0.0; // dot( eH, eP )
            double U4 = (do_base_transform) ? Hx * Qx + Hy * Qy + Hz * Qz : 1.0; // dot( eH, eQ )

            // Calculate polarization transfer matrix
            // Note: eTE = perpendicular to face normal vector = Horizontal polarization
            //       eTM = parallel to face normal vector = Vertical polarization
            double amplitude = std::sqrt(gain); // Reduction in amplitude caused by conductive medium
            if (interaction_type == 1)          // Scale amplitude in case of transmission
                amplitude *= std::sqrt((1.0 - reflection_gain) / refraction_gain);

            double VV_Re = amplitude * (U1 * Q1 * eTM_Re + U3 * Q2 * eTE_Re),
                   VV_Im = amplitude * (U1 * Q1 * eTM_Im + U3 * Q2 * eTE_Im),
                   HV_Re = amplitude * (U2 * Q1 * eTM_Re + U4 * Q2 * eTE_Re),
                   HV_Im = amplitude * (U2 * Q1 * eTM_Im + U4 * Q2 * eTE_Im),
                   VH_Re = amplitude * (U1 * Q3 * eTM_Re + U3 * Q4 * eTE_Re),
                   VH_Im = amplitude * (U1 * Q3 * eTM_Im + U3 * Q4 * eTE_Im),
                   HH_Re = amplitude * (U2 * Q3 * eTM_Re + U4 * Q4 * eTE_Re),
                   HH_Im = amplitude * (U2 * Q3 * eTM_Im + U4 * Q4 * eTE_Im);

            // Write XPRMAT
            p_xprmatN[i_rayN] = (dtype)VV_Re;
            p_xprmatN[i_rayN + n_rayN_t] = (dtype)VV_Im;
            p_xprmatN[i_rayN + 2 * n_rayN_t] = (dtype)HV_Re;
            p_xprmatN[i_rayN + 3 * n_rayN_t] = (dtype)HV_Im;
            p_xprmatN[i_rayN + 4 * n_rayN_t] = (dtype)VH_Re;
            p_xprmatN[i_rayN + 5 * n_rayN_t] = (dtype)VH_Im;
            p_xprmatN[i_rayN + 6 * n_rayN_t] = (dtype)HH_Re;
            p_xprmatN[i_rayN + 7 * n_rayN_t] = (dtype)HH_Im;
        }

        // Write trivecN
        if (use_ray_tube && p_trivecN != nullptr)
        {
            p_trivecN[i_rayN] = (dtype)p_trivec_tmp[0];
            p_trivecN[i_rayN + n_rayN_t] = (dtype)p_trivec_tmp[1];
            p_trivecN[i_rayN + 2 * n_rayN_t] = (dtype)p_trivec_tmp[2];
            p_trivecN[i_rayN + 3 * n_rayN_t] = (dtype)p_trivec_tmp[3];
            p_trivecN[i_rayN + 4 * n_rayN_t] = (dtype)p_trivec_tmp[4];
            p_trivecN[i_rayN + 5 * n_rayN_t] = (dtype)p_trivec_tmp[5];
            p_trivecN[i_rayN + 6 * n_rayN_t] = (dtype)p_trivec_tmp[6];
            p_trivecN[i_rayN + 7 * n_rayN_t] = (dtype)p_trivec_tmp[7];
            p_trivecN[i_rayN + 8 * n_rayN_t] = (dtype)p_trivec_tmp[8];
        }

        // Write tridirN
        if (use_ray_tube == 1 && p_tridirN != nullptr)
        {
            p_tridirN[i_rayN] = (dtype)p_tridir_tmp[0];
            p_tridirN[i_rayN + n_rayN_t] = (dtype)p_tridir_tmp[1];
            p_tridirN[i_rayN + 2 * n_rayN_t] = (dtype)p_tridir_tmp[2];
            p_tridirN[i_rayN + 3 * n_rayN_t] = (dtype)p_tridir_tmp[3];
            p_tridirN[i_rayN + 4 * n_rayN_t] = (dtype)p_tridir_tmp[4];
            p_tridirN[i_rayN + 5 * n_rayN_t] = (dtype)p_tridir_tmp[5];
        }
        else if (use_ray_tube == 2 && p_tridirN != nullptr)
        {
            p_tridirN[i_rayN] = (dtype)p_tridir_tmp[0];
            p_tridirN[i_rayN + n_rayN_t] = (dtype)p_tridir_tmp[1];
            p_tridirN[i_rayN + 2 * n_rayN_t] = (dtype)p_tridir_tmp[2];
            p_tridirN[i_rayN + 3 * n_rayN_t] = (dtype)p_tridir_tmp[3];
            p_tridirN[i_rayN + 4 * n_rayN_t] = (dtype)p_tridir_tmp[4];
            p_tridirN[i_rayN + 5 * n_rayN_t] = (dtype)p_tridir_tmp[5];
            p_tridirN[i_rayN + 6 * n_rayN_t] = (dtype)p_tridir_tmp[6];
            p_tridirN[i_rayN + 7 * n_rayN_t] = (dtype)p_tridir_tmp[7];
            p_tridirN[i_rayN + 8 * n_rayN_t] = (dtype)p_tridir_tmp[8];
        }

        // Write orig_lengthN
        if (p_orig_lengthN != nullptr)
            p_orig_lengthN[i_rayN] = (p_orig_length == nullptr) ? dtype(OF_length + ray_offset)
                                                                : dtype(p_orig_length[iRx] + OF_length + ray_offset);

        // Write fbs_angleN
        if (p_fbs_angleN != nullptr)
            p_fbs_angleN[i_rayN] = (dtype)theta;

        // Write thicknessN
        if (p_thicknessN != nullptr)
            p_thicknessN[i_rayN] = (dtype)FS_length;

        // Write edge_lengthN
        if (p_edge_lengthN != nullptr)
            p_edge_lengthN[i_rayN] = (dtype)edge_length_tmp;

        // Write normal_vecN
        if (p_normal_vecN != nullptr)
        {
            // FBS normal vector
            p_normal_vecN[i_rayN] = (dtype)Nx;
            p_normal_vecN[i_rayN + n_rayN_t] = (dtype)Ny;
            p_normal_vecN[i_rayN + 2 * n_rayN_t] = (dtype)Nz;
            p_normal_vecN[i_rayN + 3 * n_rayN_t] = (dtype)Mx;
            p_normal_vecN[i_rayN + 4 * n_rayN_t] = (dtype)My;
            p_normal_vecN[i_rayN + 5 * n_rayN_t] = (dtype)Mz;
        }

        // Write out_typeN
        if (p_out_typeN != nullptr)
            p_out_typeN[i_rayN] = (out_type != 0 && interaction_type == 2 && total_reflection) ? out_type + 1 : out_type;
    }

    // Delete ray index
    delete[] output_ray_index;
}

template void quadriga_lib::ray_mesh_interact(int interaction_type, float center_frequency,
                                              const arma::Mat<float> *orig, const arma::Mat<float> *dest, const arma::Mat<float> *fbs, const arma::Mat<float> *sbs,
                                              const arma::Mat<float> *mesh, const arma::Mat<float> *mtl_prop,
                                              const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                              const arma::Mat<float> *trivec, const arma::Mat<float> *tridir, const arma::Col<float> *orig_length,
                                              arma::Mat<float> *origN, arma::Mat<float> *destN, arma::Col<float> *gainN, arma::Mat<float> *xprmatN,
                                              arma::Mat<float> *trivecN, arma::Mat<float> *tridirN, arma::Col<float> *orig_lengthN,
                                              arma::Col<float> *fbs_angleN, arma::Col<float> *thicknessN, arma::Col<float> *edge_lengthN,
                                              arma::Mat<float> *normal_vecN, arma::s32_vec *out_typeN);

template void quadriga_lib::ray_mesh_interact(int interaction_type, double center_frequency,
                                              const arma::Mat<double> *orig, const arma::Mat<double> *dest, const arma::Mat<double> *fbs, const arma::Mat<double> *sbs,
                                              const arma::Mat<double> *mesh, const arma::Mat<double> *mtl_prop,
                                              const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                              const arma::Mat<double> *trivec, const arma::Mat<double> *tridir, const arma::Col<double> *orig_length,
                                              arma::Mat<double> *origN, arma::Mat<double> *destN, arma::Col<double> *gainN, arma::Mat<double> *xprmatN,
                                              arma::Mat<double> *trivecN, arma::Mat<double> *tridirN, arma::Col<double> *orig_lengthN,
                                              arma::Col<double> *fbs_angleN, arma::Col<double> *thicknessN, arma::Col<double> *edge_lengthN,
                                              arma::Mat<double> *normal_vecN, arma::s32_vec *out_typeN);