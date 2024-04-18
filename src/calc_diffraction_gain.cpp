// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

// FUNCTION: Number formatter
static std::string MioNum(size_t number)
{
    std::string str;
    if (number < 100000)
        str = std::to_string(number);
    else
    {
        double num = std::round(((double)number) / 1.0e4) / 100.0;
        str = std::to_string(num);
        str = num <= 100.0 ? str.substr(0, 5) : str;
        str = num <= 10.0 ? str.substr(0, 4) : str;
        str += " Mio.";
    }
    return str;
}

// FUNCTION: Calculate length
template <typename dtype>
static inline dtype calc_length(dtype Ox, dtype Oy, dtype Oz, dtype Dx, dtype Dy, dtype Dz)
{
    dtype a = Dx - Ox;
    dtype b = a * a;
    a = Dy - Oy, b += a * a;
    a = Dz - Oz, b += a * a;
    return std::sqrt(b);
}

// FUNCTION: Check if materials are the same
template <typename dtype>
static inline bool same_materials(const arma::Mat<dtype> *mtl_prop, unsigned iMa, unsigned iMb)
{
    if (iMa == iMb)
        return true;

    unsigned n_mesh = (unsigned)mtl_prop->n_rows;
    if (iMa > n_mesh || iMb > n_mesh) // Illegal state
        return false;

    unsigned iFBS = iMa - 1, iSBS = iMb - 1;
    const dtype *p_mtl_prop = mtl_prop->memptr();

    // std::cout << iFBS << " " << iSBS << std::endl;
    // std::cout << (p_mtl_prop[iFBS] == p_mtl_prop[iSBS]) << std::endl;
    // std::cout << (p_mtl_prop[iFBS + n_mesh] == p_mtl_prop[iSBS + n_mesh]) << std::endl;
    // std::cout << (p_mtl_prop[iFBS + 2 * n_mesh] == p_mtl_prop[iSBS + 2 * n_mesh]) << std::endl;
    // std::cout << (p_mtl_prop[iFBS + 3 * n_mesh] == p_mtl_prop[iSBS + 3 * n_mesh]) << std::endl;
    // std::cout << (p_mtl_prop[iFBS + 4 * n_mesh] == p_mtl_prop[iSBS + 4 * n_mesh]) << std::endl;
    // std::cout << p_mtl_prop[iFBS + 2 * n_mesh] << " " << p_mtl_prop[iSBS + 2 * n_mesh] << std::endl;

    if (p_mtl_prop[iFBS] == p_mtl_prop[iSBS] &&
        p_mtl_prop[iFBS + n_mesh] == p_mtl_prop[iSBS + n_mesh] &&
        p_mtl_prop[iFBS + 2 * n_mesh] == p_mtl_prop[iSBS + 2 * n_mesh] &&
        p_mtl_prop[iFBS + 3 * n_mesh] == p_mtl_prop[iSBS + 3 * n_mesh] &&
        p_mtl_prop[iFBS + 4 * n_mesh] == p_mtl_prop[iSBS + 4 * n_mesh])
        return true;

    return false;
}

// FUNCTION: Calculate the in-medium gain
template <typename dtype>
static inline dtype medium_gain_linear(const arma::Mat<dtype> *mtl_prop, unsigned iM, dtype dist, dtype fGHz)
{
    // Input variables:
    //  mtl_prop        Material properties (entire matrix)
    //  iM              Index of the material, 1-based
    //  dist            Length of the path inside the medium
    //  fGHz            Frequency in GHz

    unsigned no_mesh = (unsigned)mtl_prop->n_elem;
    if (iM == 0 || iM > no_mesh) // Illegal state
        return (dtype)0.0;

    unsigned iM0 = iM - 1;
    dtype eta_r = mtl_prop->at(iM0, 0) * std::pow(fGHz, mtl_prop->at(iM0, 1));    // Relative permittivity (real part), Rec. ITU-R P.2040-1, eq. 28
    dtype sigma = mtl_prop->at(iM0, 2) * std::pow(fGHz, mtl_prop->at(iM0, 3));    // Conductivity, Rec. ITU-R P.2040-1, eq. 29
    dtype eta_i = sigma * (dtype)17.98 / fGHz;                                    // Relative permittivity (imaginary part), Rec. ITU-R P.2040-1, eq. 9b
    dtype tan_delta = eta_i / eta_r;                                              // Loss tangent, Rec. ITU-R P.2040-1, eq. 13
    dtype cos_delta = (dtype)1.0 / std::sqrt((dtype)1.0 + tan_delta * tan_delta); // Trigonometric identity

    // Attenuation distance at which the field amplitude falls by 1/e, ITU-R P.2040-1, eq. 23a
    dtype Delta = (dtype)2.0 * cos_delta / ((dtype)1.0 - cos_delta);
    Delta = std::sqrt(Delta) * (dtype)0.0477135 / (fGHz * std::sqrt(eta_r));

    dtype A = (dtype)8.686 / Delta;             // Attenuation in db/m, ITU-R P.2040-1, eq. 26
    A *= dist;                                  // Total attenuation in dB
    A = std::pow((dtype)10.0, (dtype)-0.1 * A); // Linear gain
    return A;
}

// FUNCTION: Calculate the medium-to-medium transition gain
template <typename dtype>
static inline dtype transition_gain_linear(const arma::Mat<dtype> *mtl_prop, unsigned iMa, unsigned iMb, dtype theta, dtype fGHz)
{
    // Input variables:
    //  mtl_prop        Material properties (entire matrix)
    //  iMa             Index of the material 1, 1-based, 0 = Air
    //  iMb             Index of the material 2, 1-based, 0 = Air
    //  theta           Incidence angle at material 1 (negative for i-i)
    //  fGHz            Frequency in GHz

    unsigned n_mesh = (unsigned)mtl_prop->n_rows;
    if (iMa > n_mesh || iMb > n_mesh) // Illegal state
        return (dtype)0.0;

    unsigned iFBS = iMa - 1, iSBS = iMb - 1;
    const dtype *p_mtl_prop = mtl_prop->memptr();

    // Convert to double
    double dTheta = (double)theta;

    // Limit value to 0 ... 1 for calculating reflection and transmission coefficients
    double abs_cos_theta = std::abs(std::cos(dTheta + 1.570796326794897));
    abs_cos_theta = (abs_cos_theta > 1.0) ? 1.0 : abs_cos_theta;
    double sin_theta = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta); // Trigonometric identity

    // Select the properties of the two materials
    double kR1 = 1.0, kR2 = 0.0, kR3 = 0.0, kR4 = 0.0; // First material properties : air
    double kS1 = 1.0, kS2 = 0.0, kS3 = 0.0, kS4 = 0.0; // Second material properties : air
    double transition_gain = 1.0;                      // Additional gain for face transition, linear scale

    if (iMa != 0)
    {
        if (dTheta >= 0.0) // Ray hits front side of FBS/SBS face, set second material to object material
        {
            kS1 = (double)p_mtl_prop[iFBS];
            kS2 = (double)p_mtl_prop[iFBS + n_mesh];
            kS3 = (double)p_mtl_prop[iFBS + 2 * n_mesh];
            kS4 = (double)p_mtl_prop[iFBS + 3 * n_mesh];
            transition_gain = std::pow(10.0, -0.1 * (double)p_mtl_prop[iFBS + 4 * n_mesh]);
        }
        else // Ray hits back side of FBS face, set first material to object material
        {
            kR1 = (double)p_mtl_prop[iFBS];
            kR2 = (double)p_mtl_prop[iFBS + n_mesh];
            kR3 = (double)p_mtl_prop[iFBS + 2 * n_mesh];
            kR4 = (double)p_mtl_prop[iFBS + 3 * n_mesh];
        }
    }

    if (iMb != 0) // Material to material transition
    {
        if (dTheta >= 0.0) // SBS (front side) is hit first
        {
            kR1 = (double)p_mtl_prop[iSBS];
            kR2 = (double)p_mtl_prop[iSBS + n_mesh];
            kR3 = (double)p_mtl_prop[iSBS + 2 * n_mesh];
            kR4 = (double)p_mtl_prop[iSBS + 3 * n_mesh];
        }
        else // FBS (back side) is hit first
        {
            kS1 = (double)p_mtl_prop[iSBS];
            kS2 = (double)p_mtl_prop[iSBS + n_mesh];
            kS3 = (double)p_mtl_prop[iSBS + 2 * n_mesh];
            kS4 = (double)p_mtl_prop[iSBS + 3 * n_mesh];
            transition_gain = std::pow(10.0, -0.1 * (double)p_mtl_prop[iSBS + 4 * n_mesh]);
        }
    }

    // Calculate complex-valued relative permittivity of medium 1 and 2, ITU-R P.2040-1, eq. (9b)
    double scl = -17.98 / (double)fGHz;
    std::complex<double> eta1(kR1 * std::pow(fGHz, kR2), scl * kR3 * std::pow(fGHz, kR4)); // Material 1
    std::complex<double> eta2(kS1 * std::pow(fGHz, kS2), scl * kS3 * std::pow(fGHz, kS4)); // Material 2
    bool dense_to_light = std::real(eta1) > std::real(eta2);

    double reflection_gain = 0.0;
    if (!dense_to_light)
    {
        // Calculate cos_theta2 from Rec. ITU-R P.2040-1, eq. (33)
        std::complex<double> eta1_div_eta2 = eta1 / eta2; // Complex division
        std::complex<double> cos_theta2 = std::sqrt(1.0 - eta1_div_eta2 * sin_theta * sin_theta);

        // Calculate sqrt(eta1) and sqrt(eta2) needed for ITU-R P.2040-1, eq. (31) and (32)
        eta1 = std::sqrt(eta1); // Complex square root
        eta2 = std::sqrt(eta2); // Complex square root

        // Calculate Reflection coefficients  ITU-R P.2040-1, eq. (31)
        std::complex<double> R_eTE = 0.0, R_eTM = 0.0;

        R_eTE = (eta1 * abs_cos_theta - eta2 * cos_theta2) / (eta1 * abs_cos_theta + eta2 * cos_theta2),
        R_eTM = (eta2 * abs_cos_theta - eta1 * cos_theta2) / (eta2 * abs_cos_theta + eta1 * cos_theta2),
        reflection_gain = 0.5 * (std::norm(R_eTE) + std::norm(R_eTM));
    }

    return dtype(transition_gain * (1.0 - reflection_gain));
}

template <typename dtype>
void quadriga_lib::calc_diffraction_gain(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest,
                                         const arma::Mat<dtype> *mesh, const arma::Mat<dtype> *mtl_prop,
                                         dtype center_frequency, int lod,
                                         arma::Col<dtype> *gain, arma::Cube<dtype> *coord, int verbose,
                                         const arma::Col<unsigned> *sub_mesh_index)
{
    // Ray offset is used to detect co-location of points, value in meters
    const dtype ray_offset = (dtype)0.001;

    // Check for NULL pointers
    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (dest == nullptr)
        throw std::invalid_argument("Input 'dest' cannot be NULL.");
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mtl_prop == nullptr)
        throw std::invalid_argument("Input 'mtl_prop' cannot be NULL.");

    // Check for correct number of columns
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    if (dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mtl_prop->n_cols != 5)
        throw std::invalid_argument("Input 'mtl_prop' must have 5 columns.");

    const arma::uword n_pos = orig->n_rows;  // Number of positions
    const arma::uword n_mesh = mesh->n_rows; // Number of mesh elements
    const size_t n_pos_t = (size_t)n_pos;    // Number of positions as size_t

    // Check for correct number of rows
    if (dest->n_rows != n_pos)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");
    if (mtl_prop->n_rows != n_mesh)
        throw std::invalid_argument("Number of rows in 'mesh' and 'mtl_prop' dont match.");

    // Frequency in GHz
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    dtype fGHz = center_frequency * (dtype)1.0e-9;

    // Check range of LOD
    if ((unsigned)lod > 6U)
        throw std::invalid_argument("Input 'lod' must have values in the range 0-6.");

    // Generate diffraction paths
    arma::Cube<dtype> ray_x, ray_y, ray_z, weight;
    if (lod == 0)
        weight.ones(n_pos, 1, 1);
    else
        quadriga_lib::generate_diffraction_paths<dtype>(orig, dest, center_frequency, lod, &ray_x, &ray_y, &ray_z, &weight);

    // Dimensions of the diffraction ellipsoid
    const size_t n_path_t = (size_t)weight.n_cols;
    const size_t n_seg_t = (size_t)weight.n_slices;
    const size_t n_ray_t = n_pos_t * n_path_t;

    if (n_path_t > 61) // Just to be sure for future updates
        throw std::invalid_argument("Max. number of paths is currently fixed to 61.");

    // Track the state of each path:
    unsigned *p_ray_state = new unsigned[n_ray_t]();       // Current state, 0 = outside, otherwise index of last interaction
    unsigned *p_next_transition = new unsigned[n_ray_t](); // Next transition buffer (for overlapping mesh)

    // Pointer to the path weights
    dtype *p_weight = weight.memptr();

    if (verbose)
        std::cout << "Estimating diffraction gain with " << n_path_t << " paths * "
                  << n_seg_t << " segments for " << MioNum(n_pos_t) << " positions." << std::endl;

    // Test if diffraction paths are blocked - segment by segment
    for (size_t iS = 0; iS < n_seg_t; ++iS)
    {
        // Obtain the origin points of the current segment
        arma::Mat<dtype> s_orig; // Origin of paths for the current segment
        if (iS == 0)
            s_orig = arma::repmat(*orig, n_path_t, 1);
        else
        {
            s_orig.set_size(n_ray_t, 3);
            dtype *p_orig = s_orig.memptr();
            size_t no_bytes = n_ray_t * sizeof(dtype);
            std::memcpy(p_orig, ray_x.slice_memptr(iS - 1), no_bytes);
            std::memcpy(&p_orig[n_ray_t], ray_y.slice_memptr(iS - 1), no_bytes);
            std::memcpy(&p_orig[2 * n_ray_t], ray_z.slice_memptr(iS - 1), no_bytes);
        }

        // Obtain the destination points of the current segment
        arma::Mat<dtype> s_dest; // Destination of paths for the current segment
        if (iS == n_seg_t - 1)
            s_dest = arma::repmat(*dest, n_path_t, 1);
        else
        {
            s_dest.set_size(n_ray_t, 3);
            dtype *p_dest = s_dest.memptr();
            size_t no_bytes = n_ray_t * sizeof(dtype);
            std::memcpy(p_dest, ray_x.slice_memptr(iS), no_bytes);
            std::memcpy(&p_dest[n_ray_t], ray_y.slice_memptr(iS), no_bytes);
            std::memcpy(&p_dest[2 * n_ray_t], ray_z.slice_memptr(iS), no_bytes);
        }

        // Build global ray index for the current segment
        arma::Col<size_t> s_iRAY = arma::regspace<arma::Col<size_t>>(0, n_ray_t - 1);
        size_t n_ray_r = n_ray_t; // Number of rays in reduced set (starts with all rays)

        // Check which rays have been discontinued in a previous segment
        if (iS != 0) // Only for second segment and onwards
        {
            // Allocate memory for continued rays start and end points
            arma::Mat<dtype> c_orig(n_ray_t, 3, arma::fill::none);
            arma::Mat<dtype> c_dest(n_ray_t, 3, arma::fill::none);
            arma::Col<size_t> c_iRAY(n_ray_t); // New ray index

            size_t n_continue = 0;
            size_t previous_segment_ind = (iS - 1) * n_ray_t;
            for (size_t iR = 0; iR < n_ray_r; ++iR) // Iterate through all rays
            {
                size_t iG = s_iRAY.at(iR);
                dtype power = p_weight[previous_segment_ind + iG];
                if (power > (dtype)1.0e-20) // Continue ray
                {
                    size_t iC = n_continue++;
                    c_orig.at(iC, 0) = s_orig.at(iR, 0), c_orig.at(iC, 1) = s_orig.at(iR, 1), c_orig.at(iC, 2) = s_orig.at(iR, 2);
                    c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                    c_iRAY.at(iC) = iG;
                }
                else // Set current segment power to 0 as well
                    p_weight[iS * n_ray_t + iG] = (dtype)0.0;
            }

            // Create reduced set of rays
            if (n_continue < n_ray_t)
            {
                s_orig = arma::resize(c_orig, n_continue, 3);
                s_dest = arma::resize(c_dest, n_continue, 3);
                s_iRAY = arma::resize(c_iRAY, n_continue, 1);
                n_ray_r = n_continue;
            }
        }

        // Trace the rays of the current segment. Find where they are blocked by objects.
        // Calculate losses caused by materials until destination point is reached.

        while (n_ray_r > 0) // Run until there is no ray left to trace
        {
            if (verbose) // Debug output
                std::cout << "  Seg. " << iS << " : " << MioNum(n_ray_r) << " rays" << std::flush;

            // Calculate interaction points of rays and 3D mesh
            arma::Mat<dtype> fbs, sbs;
            arma::Col<unsigned> no_interact, fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect<dtype>(&s_orig, &s_dest, mesh, &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind, sub_mesh_index);

            // Pointers
            unsigned *p_no_interact = no_interact.memptr(); // Pointer to 'no_interact'
            unsigned *p_fbs_ind = fbs_ind.memptr();         // Pointer to 'fbs_ind'
            unsigned *p_sbs_ind = sbs_ind.memptr();         // Pointer to 'fbs_ind'

            // Create hit index
            size_t no_mesh_hit = 0;                    // Number of mesh-hits
            size_t *p_hit_ind = new size_t[n_ray_r](); // Hit index, initialized to 0
            for (size_t iR = 0; iR < n_ray_r; ++iR)    // Iterate through all rays
                if (p_fbs_ind[iR] != 0U)
                    p_hit_ind[iR] = no_mesh_hit++;

            if (verbose) // Debug output
                std::cout << ", " << MioNum(no_mesh_hit) << " mesh hits" << std::flush;
            if (verbose == 2) // Debug output
                std::cout << std::endl;

            // Calculate transmission gain and in-medium loss
            // - Outputs are only generated when mesh was hit, (origN.n_rows <= s_orig.n_rows)
            arma::Mat<dtype> origN;      // New origin after transmission
            arma::Col<dtype> gainN;      // Transmission gain
            arma::Col<dtype> fbs_angleN; // Incidence angle at FBS
            arma::Col<int> typeN;        // Medium to medium transition indicator

            if (no_mesh_hit != 0)
                quadriga_lib::ray_mesh_interact<dtype>(1, center_frequency, &s_orig, &s_dest, &fbs, &sbs, mesh, mtl_prop,
                                                       &fbs_ind, &sbs_ind, nullptr, nullptr, nullptr, &origN, nullptr,
                                                       &gainN, nullptr, nullptr, nullptr, nullptr, &fbs_angleN, nullptr,
                                                       nullptr, nullptr, &typeN);

            // Pointers
            dtype *p_gainN = gainN.memptr(); // Pointer to 'gainN'
            int *p_typeN = typeN.memptr();   // Pointer to 'typeN'

            // Count double and multi-interactions
            size_t n_continue = 0;                  // Counter for continued rays
            for (size_t iR = 0; iR < n_ray_r; ++iR) // Iterate through all rays
                if (p_no_interact[iR] > 1)
                    ++n_continue;

            // Allocate memory for multi-hit start and end points
            arma::Mat<dtype> c_orig(n_continue, 3, arma::fill::none);
            arma::Mat<dtype> c_dest(n_continue, 3, arma::fill::none);
            arma::Col<size_t> c_iRAY(n_continue); // New ray index

            // Update path weights, taking material effects into account
            n_continue = 0;                         // Rest counter
            for (size_t iR = 0; iR < n_ray_r; ++iR) // Iterate through all rays
            {
                unsigned nH = p_no_interact[iR]; // Number of mesh-hits between "orig" and "dest"
                unsigned iM1 = p_fbs_ind[iR];    // Material index of FBS, 1-based
                unsigned iM2 = p_sbs_ind[iR];    // Material index of SBS, 1-based

                size_t iG = s_iRAY.at(iR);                 // Ray index in global set
                unsigned RS = p_ray_state[iG];             // Ray state, 0 = outside, otherwise material index of current material
                unsigned NT = p_next_transition[iG];       // Next transition buffer, usually 0 unless mesh overlap
                dtype power = p_weight[iS * n_ray_t + iG]; // Current segment weight

                size_t iH = p_hit_ind[iR];                                     // Ray index in reduced set
                int typeH = (nH == 0) ? 0 : p_typeN[iH];                       // Hit-type
                dtype interaction_gain = (nH == 0) ? (dtype)1.0 : p_gainN[iH]; // Gain calculated by "quadriga_lib::ray_mesh_interact"

                if (nH == 0 && RS != 0) // Entire ray is inside the object
                {
                    dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                    power *= medium_gain_linear(mtl_prop, RS, dist, fGHz);
                }
                else if (nH == 0) // Entire ray is outside
                {                 // Nothing changes
                }
                else if ((nH == 1 && typeH == 1) || // Single hit, o-i transition
                         (nH == 2 && typeH == 7) || // Double hit at overlapping faces, oo-ii
                         (nH == 2 && typeH == 13))  // Edge Hit, o-i
                {
                    if (RS == 0) // State (o) + o-i transition
                    {
                        dtype dist = calc_length(fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                        dist = dist > ray_offset ? dist - ray_offset : dist;
                        power *= interaction_gain * medium_gain_linear(mtl_prop, iM1, dist, fGHz);
                        p_ray_state[iG] = iM1; // Change state to inside
                    }
                    else // State (i) + o-i transition, Overlapping mesh
                    {
                        // Ignore o-i transition @ FBS, Entire ray is inside material 1
                        dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                        power *= medium_gain_linear(mtl_prop, RS, dist, fGHz);
                        p_next_transition[iG] = iM1;
                    }
                }
                else if ((nH == 1 && typeH == 2) || // Single hit, i-o transition
                         (nH == 2 && typeH == 8) || // Double hit at overlapping faces, ii-oo
                         (nH == 2 && typeH == 14))  // Edge Hit, i-o
                {
                    if (RS == 0) // State (o) + i-o transition
                    {            // False inside state
                        power *= interaction_gain;
                    }
                    else if (NT == 0) // No material in buffer, State (i) + i-o transition
                    {
                        power *= interaction_gain; // Medium loss already included in interaction gain
                        p_ray_state[iG] = 0;       // Change state to outside
                    }
                    else if (nH == 1) // Material is in buffer, State (i) + virtual i-i transition
                    {
                        if (same_materials(mtl_prop, NT, iM1)) // Entire M2 is embedded inside M1, Ignore M2 completely!
                        {
                            dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                            power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS
                            p_next_transition[iG] = 0;                             // Clear next transition buffer
                        }
                        else
                        {
                            dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                            power *= medium_gain_linear(mtl_prop, RS, dist, fGHz);
                            power *= transition_gain_linear(mtl_prop, RS, NT, fbs_angleN.at(iH), fGHz);
                            dist = calc_length(fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                            power *= medium_gain_linear(mtl_prop, NT, dist, fGHz);
                            p_ray_state[iG] = NT;      // Set ray state to NT (inside state)
                            p_next_transition[iG] = 0; // Clear next transition buffer
                        }
                    }
                    else if (nH == 2) // Material is in buffer, Double hit, State (i) + virtual ii-oo transition from M1 to Air
                    {
                        dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                        power *= medium_gain_linear(mtl_prop, RS, dist, fGHz);
                        power *= transition_gain_linear(mtl_prop, RS, 0, fbs_angleN.at(iH), fGHz);
                        p_ray_state[iG] = 0;       // Change state to outside
                        p_next_transition[iG] = 0; // Clear next transition buffer
                    }
                }
                else if (nH == 2 && typeH == 1)
                {
                    if (RS == 0) // Double hit, (o) + o-i-o
                    {
                        power *= interaction_gain; // Include o-i transition loss @ FBS
                        p_ray_state[iG] = iM1;     // Change state to inside

                        // Add path to next iteration
                        if (power > (dtype)1.0e-20)
                        {
                            size_t iC = n_continue++;
                            c_orig.at(iC, 0) = origN.at(iH, 0), c_orig.at(iC, 1) = origN.at(iH, 1), c_orig.at(iC, 2) = origN.at(iH, 2);
                            c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                            c_iRAY.at(iC) = iG;
                        }
                    }
                    else // Double hit, (i) + o-i-o, overlapping mesh
                    {
                        dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                        power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS
                        p_next_transition[iG] = iM1;

                        // Add path to next iteration
                        if (power > (dtype)1.0e-20)
                        {
                            size_t iC = n_continue++;
                            c_orig.at(iC, 0) = origN.at(iH, 0), c_orig.at(iC, 1) = origN.at(iH, 1), c_orig.at(iC, 2) = origN.at(iH, 2);
                            c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                            c_iRAY.at(iC) = iG;
                        }
                    }
                }
                else if (nH == 2 && typeH == 2) // Double hit, i-o-i
                {                               // Medium loss of first inside segment already included in interaction gain, segment 2 must be added
                    if (NT == 0)                // No buffer, state (i/o) + i-o-i transition
                    {
                        if (iM2 == 0)           // No material for SBS given, illegal state
                            power = (dtype)0.0; // Terminate ray
                        else
                        {
                            power *= interaction_gain; // Medium loss of first inside segment already included in interaction gain
                            p_ray_state[iG] = 0;       // Change state to outside

                            // Add path to next iteration
                            if (power > (dtype)1.0e-20)
                            {
                                size_t iC = n_continue++;
                                c_orig.at(iC, 0) = origN.at(iH, 0), c_orig.at(iC, 1) = origN.at(iH, 1), c_orig.at(iC, 2) = origN.at(iH, 2);
                                c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                                c_iRAY.at(iC) = iG;
                            }
                        }
                    }
                    else if (RS != 0) // Ray state in buffer, state (i) + virtual i-i + i-o
                    {
                        dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                        if (same_materials(mtl_prop, NT, iM1)) // Entire M2 is embedded inside M1, Ignore M2 completely!
                        {
                            // Add in-medium loss for medium 1 (from Orig to Dest)
                            power *= medium_gain_linear(mtl_prop, RS, dist + ray_offset, fGHz); // M1 medium defined by RS
                            p_next_transition[iG] = 0;                                          // Clear next transition buffer
                        }
                        else
                        {
                            power *= medium_gain_linear(mtl_prop, RS, dist, fGHz);
                            power *= transition_gain_linear(mtl_prop, RS, NT, fbs_angleN.at(iH), fGHz);
                            power *= medium_gain_linear(mtl_prop, NT, ray_offset, fGHz);
                            p_ray_state[iG] = NT;      // Set ray state to NT (inside state)
                            p_next_transition[iG] = 0; // Clear next transition buffer
                        }

                        // Add path to next iteration
                        if (power > (dtype)1.0e-20)
                        {
                            size_t iC = n_continue++;
                            c_orig.at(iC, 0) = origN.at(iH, 0), c_orig.at(iC, 1) = origN.at(iH, 1), c_orig.at(iC, 2) = origN.at(iH, 2);
                            c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                            c_iRAY.at(iC) = iG;
                        }
                    }
                    else
                        power = (dtype)0.0; // Terminate ray
                }
                else if (nH == 2 && (typeH == 4 || typeH == 5)) // i-i transition
                {
                    if (RS == 0)            // State (o) + i-i transition, should not happen
                        power = (dtype)0.0; // Terminate ray
                    else if (NT == 0)       // No buffer, state (i) + i-i transition
                    {
                        if (iM1 == 0 || iM2 == 0) // Missing material, illegal state
                            power = (dtype)0.0;   // Terminate ray
                        else
                        {
                            // Medium loss of segment 1 already included in interaction_gain, segment 2 must be added
                            unsigned iM = (typeH == 5) ? iM2 : iM1; // If back wall was hit first, use SBS material
                            dtype dist = calc_length(fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                            power *= interaction_gain * medium_gain_linear(mtl_prop, iM, dist - ray_offset, fGHz);
                            p_ray_state[iG] = iM; // Update ray state
                        }
                    }
                    else if (NT != 0)
                    {
                        // Continue in medium defined by RS
                        dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                        power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS

                        // Swap the material stored in NT
                        p_next_transition[iG] = same_materials(mtl_prop, NT, iM1) ? iM2 : iM1;
                    }
                    else                    // Ray state in buffer
                        power = (dtype)0.0; // Terminate ray
                }
                else if (nH == 2 && typeH == 10) // Edge Hit, o-i-o
                {
                    if (RS == 0)
                    {
                        power *= interaction_gain;
                        p_ray_state[iG] = 0; // Set state to outside
                    }
                    else // Ignore interaction, keep ray state
                    {
                        if (same_materials(mtl_prop, iM1, iM2)) // Ignore hit
                        {
                            // Add in-medium loss for medium 1 (from Orig to Dest)
                            dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                            power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS
                        }
                        else // Add i-i transition
                        {
                            dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                            power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS
                            power *= transition_gain_linear(mtl_prop, RS, iM1, fbs_angleN.at(iH), fGHz);
                            dist = calc_length(fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                            power *= medium_gain_linear(mtl_prop, iM1, dist, fGHz); // M1 medium defined by iM1
                            p_ray_state[iG] = iM1;                                  // Continue in M1
                        }
                    }
                }
                else if (nH == 2 && typeH == 11) // Edge Hit, i-o-i
                {
                    if (RS == 0) // Outdoor state
                    {
                        power *= interaction_gain;

                        // Account for floating point precision
                        dtype dist = calc_length(fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2), sbs.at(iR, 0), sbs.at(iR, 1), sbs.at(iR, 2));
                        p_ray_state[iG] = (dist > 1.0e-6) ? iM2 : 0; // Continue inside material defined by SBS
                    }
                    else // Indoor state
                    {
                        if (same_materials(mtl_prop, iM1, iM2)) // Ignore hit
                        {
                            // Add in-medium loss for medium 1 (from Orig to Dest)
                            dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                            power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS
                        }
                        else // Add i-i transition
                        {
                            // Medium loss of segment 1 already included in interaction_gain
                            dtype dist = calc_length(fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2), s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                            power *= interaction_gain * medium_gain_linear(mtl_prop, iM2, dist - ray_offset, fGHz);
                            p_ray_state[iG] = iM2; // Update ray state
                        }
                    }
                }
                else if (nH > 2) // Multi-hit
                {
                    if (RS == 0) // Outside state
                    {
                        if (NT != 0)                       // Cannot have i-i transition in buffer
                            power = (dtype)0.0;            // Terminate ray
                        else if (typeH == 1 || typeH == 7) // State (o) + o-i transition
                        {
                            power *= interaction_gain; // Add transition gain
                            p_ray_state[iG] = iM1;     // Change state to inside M1
                        }
                        else if (typeH == 2)           // False inside state
                            power *= interaction_gain; // Add transition and medium gain
                        else if (typeH == 10)          // Edge hit, State (o) + o-i-o transition
                            power *= interaction_gain; // Add transition gain, leave outside state
                        else if (typeH == 13)
                        {
                            power *= interaction_gain; // Add transition gain
                            p_ray_state[iG] = iM1;     // Change state to inside M1
                            p_next_transition[iG] = iM2;
                        }
                        else                    // Some other hit type
                            power = (dtype)0.0; // Terminate ray
                    }
                    else // Inside state
                    {
                        if (typeH == 1 || typeH == 7 || typeH == 13) // State (i) + o-i transition, Overlapping mesh
                        {
                            dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                            power *= medium_gain_linear(mtl_prop, RS, dist + ray_offset, fGHz);
                            p_next_transition[iG] = iM1;
                        }
                        else if (typeH == 2 || typeH == 14) // State (i) + i-o transition
                        {
                            if (NT == 0) // No material in buffer, State (i) + i-o transition
                            {
                                power *= interaction_gain; // Medium loss already included in interaction gain
                                p_ray_state[iG] = 0;       // Change state to outside
                            }
                            else // Material is in buffer, State (i) + virtual i-i transition
                            {
                                dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                                if (same_materials(mtl_prop, NT, iM1)) // Entire M2 is embedded inside M1, Ignore M2 completely!
                                {
                                    // Add in-medium loss for medium 1 (from Orig to Dest)
                                    power *= medium_gain_linear(mtl_prop, RS, dist + ray_offset, fGHz); // M1 medium defined by RS
                                    p_next_transition[iG] = 0;                                          // Clear next transition buffer
                                }
                                else
                                {
                                    power *= medium_gain_linear(mtl_prop, RS, dist, fGHz);
                                    power *= transition_gain_linear(mtl_prop, RS, NT, fbs_angleN.at(iH), fGHz);
                                    power *= medium_gain_linear(mtl_prop, NT, ray_offset, fGHz);
                                    p_ray_state[iG] = NT;      // Set ray state to NT (inside state)
                                    p_next_transition[iG] = 0; // Clear next transition buffer
                                }
                            }
                        }
                        else if (typeH == 4 || typeH == 5) // State (i) + i-i transition
                        {
                            if (NT != 0)                   // Material in buffer (should not happen)
                            {                              // Probably false detection in previous interaction
                                power *= interaction_gain; // Add transition and in-medium gain
                                p_next_transition[iG] = 0; // Clear next transition buffer
                            }
                            else
                            {
                                power *= interaction_gain;
                                p_ray_state[iG] = (typeH == 5) ? iM2 : iM1;
                            }
                        }
                        else if (typeH == 8) // Overlapping faces
                        {
                            if (NT == 0) // No material in buffer, State (i) + ii-oo transition
                            {
                                power *= interaction_gain; // Medium loss already included in interaction gain
                                p_ray_state[iG] = 0;       // Change state to outside
                            }
                            else // Material in Buffer
                            {
                                dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                                power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS
                                power *= transition_gain_linear(mtl_prop, RS, 0, fbs_angleN.at(iH), fGHz);
                                p_ray_state[iG] = 0;       // Change state to outside
                                p_next_transition[iG] = 0; // Clear next transition buffer
                            }
                        }
                        else if (typeH == 10) // State (i) + Edge Hit, o-i-o
                        {
                            if (RS == 0) // No material in buffer, State (i/o) + i-o transition
                            {
                                power *= interaction_gain; // Medium loss already included in interaction gain
                                p_ray_state[iG] = 0;       // Change state to outside
                            }
                            else if (NT == 0)
                            {
                                dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                                if (same_materials(mtl_prop, iM1, iM2)) // Ignore hit
                                {
                                    // Add in-medium loss for medium 1 (from Orig to Dest)
                                    power *= medium_gain_linear(mtl_prop, RS, dist + ray_offset, fGHz); // M1 medium defined by RS
                                }
                                else // Add i-i transition
                                {
                                    power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS
                                    power *= transition_gain_linear(mtl_prop, RS, iM1, fbs_angleN.at(iH), fGHz);
                                    power *= medium_gain_linear(mtl_prop, iM1, ray_offset, fGHz); // M1 medium defined by iM1
                                    p_ray_state[iG] = iM1;                                        // Continue in M1
                                }
                            }
                            else if (NT != 0) // Material is in buffer, State (i) + virtual i-i transition
                            {
                                dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                                if (same_materials(mtl_prop, NT, iM1)) // Entire M2 is embedded inside M1, Ignore M2 completely!
                                {
                                    power *= medium_gain_linear(mtl_prop, RS, dist + ray_offset, fGHz); // M1 medium defined by RS
                                    p_next_transition[iG] = 0;                                          // Clear next transition buffer
                                }
                                else
                                {
                                    power *= medium_gain_linear(mtl_prop, RS, dist, fGHz); // M1 medium defined by RS
                                    power *= transition_gain_linear(mtl_prop, RS, NT, fbs_angleN.at(iH), fGHz);
                                    power *= medium_gain_linear(mtl_prop, NT, ray_offset, fGHz); // M2 medium defined by NT
                                    p_ray_state[iG] = NT;                                        // Set ray state to NT (inside state)
                                    p_next_transition[iG] = 0;                                   // Clear next transition buffer
                                }
                            }
                        }
                        else if (typeH == 11) // Edge Hit, i-o-i
                        {
                            if (RS == 0)
                            {
                                power *= interaction_gain;

                                // Account for floating point precision
                                dtype dist = calc_length(fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2), sbs.at(iR, 0), sbs.at(iR, 1), sbs.at(iR, 2));
                                p_ray_state[iG] = (dist > 1.0e-6) ? iM2 : 0; // Continue inside material defined by SBS
                            }
                            else if (NT == 0)
                            {
                                dtype dist = calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2), fbs.at(iR, 0), fbs.at(iR, 1), fbs.at(iR, 2));
                                if (same_materials(mtl_prop, iM1, iM2)) // Ignore hit
                                {
                                    // Add in-medium loss for medium 1 (from Orig to Dest)
                                    power *= medium_gain_linear(mtl_prop, RS, dist + ray_offset, fGHz); // M1 medium defined by RS
                                }
                                else // Add i-i transition
                                {    // Somewhat fake, since FBS and SBS normals are not aligned
                                    power *= medium_gain_linear(mtl_prop, RS, dist, fGHz);
                                    power *= transition_gain_linear(mtl_prop, RS, iM2, fbs_angleN.at(iH), fGHz);
                                    power *= medium_gain_linear(mtl_prop, iM2, ray_offset, fGHz);
                                    p_ray_state[iG] = iM2;
                                }

                                power *= medium_gain_linear(mtl_prop, RS, dist + ray_offset, fGHz); // M1 medium defined by RS
                            }
                            else // Material in buffer (should not happen)
                            {
                                power *= interaction_gain;
                                p_next_transition[iG] = 0;
                            }
                        }
                    }

                    // Add path to next iteration
                    if (power > (dtype)1.0e-20)
                    {
                        size_t iC = n_continue++;
                        c_orig.at(iC, 0) = origN.at(iH, 0), c_orig.at(iC, 1) = origN.at(iH, 1), c_orig.at(iC, 2) = origN.at(iH, 2);
                        c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                        c_iRAY.at(iC) = iG;
                    }
                }
                else // Drop ray
                    power = (dtype)0.0;

                // For debugging:
                if (verbose == 2 && n_pos == 1)
                {
                    double gg = nH == 0 ? 1.0 : interaction_gain;
                    std::cout << "nH = " << nH << ", tH = " << typeH
                              << ", RS = " << RS << " -> " << p_ray_state[iG]
                              << ", NT = " << NT << " -> " << p_next_transition[iG]
                              << ", orig = (" << s_orig.at(iR, 0) << ", " << s_orig.at(iR, 1) << ", " << s_orig.at(iR, 2) << ")"
                              << ", fbs = (" << fbs.at(iR, 0) << ", " << fbs.at(iR, 1) << ", " << fbs.at(iR, 2) << ")"
                              << ", sbs = (" << sbs.at(iR, 0) << ", " << sbs.at(iR, 1) << ", " << sbs.at(iR, 2) << ")"
                              << ", dest = (" << s_dest.at(iR, 0) << ", " << s_dest.at(iR, 1) << ", " << s_dest.at(iR, 2) << ")"
                              << ", P = (" << p_weight[iS * n_ray_t + iG] << ", " << gg << ", " << power << ")" << std::endl;
                }

                // Update segment weight with new power values
                p_weight[iS * n_ray_t + iG] = power;
            }

            if (verbose == 1) // Debug output
                std::cout << " (" << MioNum(n_continue) << " continued)" << std::endl;

            // Add multi-hits to a new launch config
            if (n_continue > 0)
            {
                s_orig = arma::resize(c_orig, n_continue, 3);
                s_dest = arma::resize(c_dest, n_continue, 3);
                s_iRAY = arma::resize(c_iRAY, n_continue, 1);
            }
            n_ray_r = n_continue;

            // Clear hit index
            delete[] p_hit_ind;
        }
    }

    // Clear ray state
    delete[] p_ray_state;
    delete[] p_next_transition;

    // Adjust size of the output containers, if needed
    const arma::uword n_seg = (arma::uword)n_seg_t - 1;

    if (gain != nullptr && gain->n_elem != n_pos)
        gain->set_size(n_pos);
    if (coord != nullptr && (coord->n_rows != 3 || coord->n_cols != n_seg || coord->n_slices != n_pos))
        coord->set_size(3, n_seg, n_pos);

    // Write output data
    dtype *p_ray_x = ray_x.memptr(), *p_ray_y = ray_y.memptr(), *p_ray_z = ray_z.memptr();
    dtype *p_gain = (gain == nullptr) ? nullptr : gain->memptr();
    dtype *p_coord = (coord == nullptr) ? nullptr : coord->memptr();

    if (p_gain != nullptr || p_coord != nullptr)
        for (size_t iR = 0; iR < n_pos_t; ++iR)
        {
            dtype scl = (dtype)0.0;
            dtype path_gain[61];

            for (size_t iP = 0; iP < n_path_t; ++iP)
            {
                dtype w = (dtype)1.0;
                size_t iG = iP * n_pos_t + iR;
                for (size_t iS = 0; iS < n_seg_t; ++iS)
                    w *= p_weight[iS * n_ray_t + iG];
                path_gain[iP] = w;
                scl += w;
            }

            if (p_gain != nullptr)
                p_gain[iR] = scl;

            if (p_coord != nullptr)
            {
                scl = (dtype)1.0 / scl;
                for (size_t iS = 0; iS < n_seg_t - 1; ++iS)
                {
                    dtype x = (dtype)0.0, y = (dtype)0.0, z = (dtype)0.0;
                    for (size_t iP = 0; iP < n_path_t; ++iP)
                    {
                        size_t iG = iS * n_pos_t * n_path_t + iP * n_pos_t + iR;
                        x += p_ray_x[iG] * path_gain[iP];
                        y += p_ray_y[iG] * path_gain[iP];
                        z += p_ray_z[iG] * path_gain[iP];
                    }
                    x *= scl, y *= scl, z *= scl;
                    *p_coord++ = x;
                    *p_coord++ = y;
                    *p_coord++ = z;
                }
            }
        }
}

template void quadriga_lib::calc_diffraction_gain(const arma::Mat<float> *orig, const arma::Mat<float> *dest,
                                                  const arma::Mat<float> *mesh, const arma::Mat<float> *mtl_prop,
                                                  float center_frequency, int lod,
                                                  arma::Col<float> *gain, arma::Cube<float> *coord, int verbose,
                                                  const arma::Col<unsigned> *sub_mesh_index);

template void quadriga_lib::calc_diffraction_gain(const arma::Mat<double> *orig, const arma::Mat<double> *dest,
                                                  const arma::Mat<double> *mesh, const arma::Mat<double> *mtl_prop,
                                                  double center_frequency, int lod,
                                                  arma::Col<double> *gain, arma::Cube<double> *coord, int verbose,
                                                  const arma::Col<unsigned> *sub_mesh_index);
