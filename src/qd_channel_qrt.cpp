// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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

#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"
#include "qrt_file_reader.hpp"

void quadriga_lib::qrt_file_parse(const std::string &fn,
                                  arma::uword *no_cir,
                                  arma::uword *no_orig,
                                  arma::uword *no_dest,
                                  arma::uvec *cir_offset,
                                  std::vector<std::string> *orig_names,
                                  std::vector<std::string> *dest_names)
{
    auto qrt = qrt_file_reader(fn);

    if (no_cir)
        *no_cir = (arma::uword)qrt.no_cir;

    if (no_orig)
        *no_orig = (arma::uword)qrt.no_orig;

    if (no_dest)
        *no_dest = (arma::uword)qrt.no_dest;

    if (qrt.cir_index.n_elem == 0 || qrt.cir_index[0] != 0)
        throw std::invalid_argument("Invalid CIR index in QRT file. Potential file corruption.");

    if (cir_offset)
    {
        cir_offset->set_size(qrt.no_dest);
        auto po = cir_offset->memptr();
        for (auto &val : qrt.cir_index)
            *po++ = (arma::uword)val;
    }

    if (dest_names)
        *dest_names = qrt.dest_names;

    if (orig_names)
    {
        orig_names->clear();
        orig_names->reserve(qrt.no_orig);
        for (arma::uword i = 0; i < qrt.no_orig; ++i)
        {
            qrt.set_orig(i);
            orig_names->push_back(qrt.orig_name);
        }
    }

    qrt.close();
}

template <typename dtype>
void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                 dtype *center_frequency, arma::Col<dtype> *tx_pos, arma::Col<dtype> *tx_orientation,
                                 arma::Col<dtype> *rx_pos, arma::Col<dtype> *rx_orientation,
                                 arma::Mat<dtype> *fbs_pos, arma::Mat<dtype> *lbs_pos,
                                 arma::Col<dtype> *path_gain, arma::Col<dtype> *path_length, arma::Mat<dtype> *M,
                                 arma::Col<dtype> *aod, arma::Col<dtype> *eod, arma::Col<dtype> *aoa, arma::Col<dtype> *eoa,
                                 std::vector<arma::Mat<dtype>> *path_coord)
{

    auto qrt = qrt_file_reader(fn, i_cir, i_orig);

    if (i_orig >= qrt.no_cir)
        throw std::invalid_argument("CIR index exceeds number of CIRs in file.");

    if (i_orig >= qrt.no_orig)
        throw std::invalid_argument("Origin (TX) index exceeds number of origin points in file.");

    double fGHz = (double)qrt.fGHz;
    double gain_at_1m = -32.45 - 20.0 * std::log10(fGHz);
    if (center_frequency)
        *center_frequency = dtype(1e9 * fGHz);

    // Positions
    dtype Ox = (dtype)qrt.orig_pos_all[0];
    dtype Oy = (dtype)qrt.orig_pos_all[1];
    dtype Oz = (dtype)qrt.orig_pos_all[2];
    if (tx_pos && downlink)
        *tx_pos = {Ox, Oy, Oz};
    if (rx_pos && !downlink)
        *rx_pos = {Ox, Oy, Oz};

    dtype Dx = (dtype)qrt.cir_pos[0];
    dtype Dy = (dtype)qrt.cir_pos[1];
    dtype Dz = (dtype)qrt.cir_pos[2];
    if (rx_pos && downlink)
        *rx_pos = {Dx, Dy, Dz};
    if (tx_pos && !downlink)
        *tx_pos = {Dx, Dy, Dz};

    // Orientations
    if (tx_orientation && downlink)
        *tx_orientation = {(dtype)qrt.orig_orientation[0],
                           (dtype)qrt.orig_orientation[1],
                           (dtype)qrt.orig_orientation[2]};

    if (tx_orientation && !downlink)
        *tx_orientation = {(dtype)qrt.cir_orientation[0],
                           (dtype)qrt.cir_orientation[1],
                           (dtype)qrt.cir_orientation[2]};

    if (rx_orientation && downlink)
        *rx_orientation = {(dtype)qrt.cir_orientation[0],
                           (dtype)qrt.cir_orientation[1],
                           (dtype)qrt.cir_orientation[2]};

    if (rx_orientation && !downlink)
        *rx_orientation = {(dtype)qrt.orig_orientation[0],
                           (dtype)qrt.orig_orientation[1],
                           (dtype)qrt.orig_orientation[2]};

    // Read polarization Matrix and interaction coordinates from file
    arma::u32_vec no_intR;
    arma::fmat xprmatR, coordR;
    unsigned no_path = qrt.read_cir(0U, (unsigned)i_cir, no_intR, xprmatR, coordR); // Note: 0U because of special constructor

    // Calculate path gain and polarization matrix
    // - xprmatR includes all interaction losses, but not the FSPL
    // - here we calculate the normalized polarization matrix M and the PG without FSPL
    if (M || path_gain)
    {
        dtype *dst = nullptr;
        if (M)
        {
            M->set_size(8, no_path);
            dst = M->memptr();
        }

        dtype *pg = nullptr;
        if (path_gain)
        {
            path_gain->set_size(no_path);
            pg = path_gain->memptr();
        }

        const float *src = xprmatR.memptr();

        for (arma::uword k = 0; k < no_path; ++k)
        {
            // load as dtype into registers
            const dtype r11 = (dtype)src[0];
            const dtype i11 = (dtype)src[1];
            const dtype r21 = (dtype)src[2];
            const dtype i21 = (dtype)src[3];
            const dtype r12 = (dtype)src[4];
            const dtype i12 = (dtype)src[5];
            const dtype r22 = (dtype)src[6];
            const dtype i22 = (dtype)src[7];

            // column powers (V and H) - path gain = max column power
            const dtype p1 = r11 * r11 + i11 * i11 + r21 * r21 + i21 * i21;
            const dtype p2 = r12 * r12 + i12 * i12 + r22 * r22 + i22 * i22;
            const dtype gain = (p1 > p2) ? p1 : p2;

            // write path gain
            if (pg)
                *pg++ = gain;

            // normalization factor: max column power -> 1
            dtype scale = (dtype)0;
            if (gain > (dtype)0)
                scale = (dtype)1 / (dtype)std::sqrt((double)gain);

            if (dst)
            {
                if (downlink) // copy, normalize
                {
                    dst[0] = r11 * scale;
                    dst[1] = i11 * scale;
                    dst[2] = r21 * scale;
                    dst[3] = i21 * scale;
                    dst[4] = r12 * scale;
                    dst[5] = i12 * scale;
                    dst[6] = r22 * scale;
                    dst[7] = i22 * scale;
                }
                else // uplink: conjugate transpose, normalize
                {
                    // H_UL = H_DL^H
                    dst[0] = r11 * scale;  // Re(h11)
                    dst[1] = -i11 * scale; // -Im(h11)
                    dst[2] = r12 * scale;  // Re(h12)
                    dst[3] = -i12 * scale; // -Im(h12)
                    dst[4] = r21 * scale;  // Re(h21)
                    dst[5] = -i21 * scale; // -Im(h21)
                    dst[6] = r22 * scale;  // Re(h22)
                    dst[7] = -i22 * scale; // -Im(h22)
                }
                dst += 8;
            }
            src += 8;
        }
    }

    bool want_angles = aod || eod || aoa || eoa;
    bool want_length = path_gain || path_length;

    // Extract path metadata
    // - here we add the FSPL from path length to the PG
    if (want_angles || want_length || fbs_pos || lbs_pos || path_coord)
    {
        // Convert interaction coordinates to desired precision (e.g. float to double)
        arma::Mat<dtype> coordD(3, coordR.n_cols, arma::fill::none);
        {
            dtype *p = coordD.memptr();
            for (auto &val : coordR)
                *p++ = (dtype)val;
        }

        // Convert path interaction coordinates into FBS/LBS positions, path length and angles
        arma::Mat<dtype> path_angles;
        arma::Col<dtype> path_length_local;

        if (want_angles && want_length)
            quadriga_lib::coord2path<dtype>(Ox, Oy, Oz, Dx, Dy, Dz, &no_intR, &coordD,
                                            &path_length_local, fbs_pos, lbs_pos, &path_angles, path_coord, !downlink);
        else if (want_angles && !want_length)
            quadriga_lib::coord2path<dtype>(Ox, Oy, Oz, Dx, Dy, Dz, &no_intR, &coordD,
                                            nullptr, fbs_pos, lbs_pos, &path_angles, path_coord, !downlink);
        else if (!want_angles && want_length)
            quadriga_lib::coord2path<dtype>(Ox, Oy, Oz, Dx, Dy, Dz, &no_intR, &coordD,
                                            &path_length_local, fbs_pos, lbs_pos, nullptr, path_coord, !downlink);
        else // want_none
            quadriga_lib::coord2path<dtype>(Ox, Oy, Oz, Dx, Dy, Dz, &no_intR, &coordD,
                                            nullptr, fbs_pos, lbs_pos, nullptr, path_coord, !downlink);

        // Adjust path gain to include the FSPL
        if (want_length)
        {
            if (path_length)
                path_length->set_size(no_path);

            dtype *src = path_length_local.memptr();
            dtype *pg = path_gain ? path_gain->memptr() : nullptr;
            dtype *pl = path_length ? path_length->memptr() : nullptr;

            for (arma::uword k = 0; k < no_path; ++k)
            {
                dtype len = src[k];
                if (pl)
                    pl[k] = len;
                if (pg)
                {
                    double gainFS = gain_at_1m - 20.0 * std::log10((double)len);
                    pg[k] *= (dtype)std::pow(10.0, 0.1 * gainFS);
                }
            }
        }

        if (aod)
        {
            aod->set_size(no_path);
            std::memcpy(aod->memptr(), path_angles.colptr(0), no_path * sizeof(dtype));
        }

        if (eod)
        {
            eod->set_size(no_path);
            std::memcpy(eod->memptr(), path_angles.colptr(1), no_path * sizeof(dtype));
        }

        if (aoa)
        {
            aoa->set_size(no_path);
            std::memcpy(aoa->memptr(), path_angles.colptr(2), no_path * sizeof(dtype));
        }

        if (eoa)
        {
            eoa->set_size(no_path);
            std::memcpy(eoa->memptr(), path_angles.colptr(3), no_path * sizeof(dtype));
        }
    }

    qrt.close();
}

template void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                          float *center_frequency, arma::Col<float> *tx_pos, arma::Col<float> *tx_orientation,
                                          arma::Col<float> *rx_pos, arma::Col<float> *rx_orientation,
                                          arma::Mat<float> *fbs_pos, arma::Mat<float> *lbs_pos,
                                          arma::Col<float> *path_gain, arma::Col<float> *path_length, arma::Mat<float> *M,
                                          arma::Col<float> *aod, arma::Col<float> *eod, arma::Col<float> *aoa, arma::Col<float> *eoa,
                                          std::vector<arma::Mat<float>> *path_coord);

template void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                          double *center_frequency, arma::Col<double> *tx_pos, arma::Col<double> *tx_orientation,
                                          arma::Col<double> *rx_pos, arma::Col<double> *rx_orientation,
                                          arma::Mat<double> *fbs_pos, arma::Mat<double> *lbs_pos,
                                          arma::Col<double> *path_gain, arma::Col<double> *path_length, arma::Mat<double> *M,
                                          arma::Col<double> *aod, arma::Col<double> *eod, arma::Col<double> *aoa, arma::Col<double> *eoa,
                                          std::vector<arma::Mat<double>> *path_coord);