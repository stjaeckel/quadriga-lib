// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// QRT ray-tracing files: qrt_file_parse, qrt_read_cache_init, then qrt_file_read with all outputs (up to two origins
// and two CIRs, both directions, both normalizations of M, one index past the end) and qrt_file_read_raw

#include "fuzz_target.hpp"

// One CIR with a LOS path and a path with two interactions, EM (8 Jones values) or scalar (2 values) per frequency
static std::vector<quadriga_lib::path> two_paths(unsigned i_cir, unsigned n_freq, bool scalar)
{
    std::vector<quadriga_lib::path> paths;
    for (unsigned n_seg : {0u, 2u})
    {
        quadriga_lib::path p(n_seg, n_freq, scalar);
        p.iC = i_cir;
        for (unsigned s = 0; s < n_seg; ++s)
        {
            float *c = p.coord(s);
            c[0] = 10.0f + (float)s, c[1] = 2.0f * (float)s, c[2] = 1.5f;
        }
        for (unsigned f = 0; f < n_freq; ++f)
        {
            float *m = p.xpr_coeff(f);
            for (unsigned k = 0; k < (scalar ? 2u : 8u); ++k)
                m[k] = 0.1f * (float)(k + 1);
        }
        if (n_seg != 0)
            p.set_interaction_type_codes(std::vector<uint8_t>(n_seg, 1));
        paths.push_back(p);
    }
    return paths;
}

void fuzz_seeds(const fz::seeds &s)
{
    s.copy("tests/data/test.qrt");
    s.copy("tests/data/test_v5.qrt");
    s.copy("tests/data/test_zero_cir.qrt");

    // Two origins, two CIRs, two receivers, two frequencies; EM and scalar layout
    for (bool scalar : {false, true})
    {
        std::string fn = s.path(scalar ? "scalar.qrt" : "em.qrt");
        arma::fvec freq = {3.5f, 28.0f};
        arma::fmat cir_pos = {{5.0f, 0.0f, 1.5f}, {6.0f, 1.0f, 1.5f}};
        arma::fmat cir_orientation = {{0.0f, 0.0f, 0.1f}, {0.0f, 0.0f, 0.2f}};
        quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orientation, {"RX0", "RX1"}, {0, 1}, 2, scalar);
        for (unsigned i_orig = 0; i_orig < 2; ++i_orig)
        {
            std::vector<quadriga_lib::path> paths = two_paths(0, 2, scalar), more = two_paths(1, 2, scalar);
            paths.insert(paths.end(), more.begin(), more.end());
            quadriga_lib::qrt_file_append<float>(fn, paths, {0.0f, (float)i_orig, 10.0f}, {0.0f, 0.0f, 0.0f},
                                                 "TX" + std::to_string(i_orig));
        }
    }
}

void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn = fz::write_file("in.qrt", data, size);

    arma::uword no_cir = 0, no_orig = 0;
    fz::clean([&]
              {
        arma::uword no_dest, no_freq;
        arma::uvec cir_offset;
        std::vector<std::string> orig_names, dest_names;
        int version;
        arma::fvec freq;
        arma::fmat cir_pos, cir_orientation, orig_pos, orig_orientation;
        quadriga_lib::qrt_file_parse(fn, &no_cir, &no_orig, &no_dest, &no_freq, &cir_offset, &orig_names, &dest_names,
                                     &version, &freq, &cir_pos, &cir_orientation, &orig_pos, &orig_orientation); });
    fz::clean([&] { quadriga_lib::qrt_read_cache_init(fn); });

    arma::uword n_orig = std::min(no_orig, (arma::uword)2), n_cir = std::min(no_cir, (arma::uword)2);
    for (arma::uword i_orig = 0; i_orig <= n_orig; ++i_orig) // i_orig == n_orig: past the end if the file has one or two
    {
        for (arma::uword i_cir = 0; i_cir <= n_cir; ++i_cir)
            for (bool downlink : {true, false})
                fz::clean([&]
                          {
                    arma::vec center_frequency, tx_pos, tx_orientation, rx_pos, rx_orientation, path_length, aod, eod, aoa, eoa;
                    arma::mat fbs_pos, lbs_pos, path_gain;
                    arma::cube M;
                    std::vector<arma::mat> path_coord;
                    arma::u32_vec no_int;
                    arma::fmat coord;
                    std::vector<uint8_t> interact_type;
                    quadriga_lib::qrt_file_read<double>(fn, i_cir, i_orig, downlink, &center_frequency, &tx_pos,
                                                        &tx_orientation, &rx_pos, &rx_orientation, &fbs_pos, &lbs_pos,
                                                        &path_gain, &path_length, &M, &aod, &eod, &aoa, &eoa,
                                                        &path_coord, downlink ? 1 : 0, &no_int, &coord, &interact_type); });
        fz::clean([&] { quadriga_lib::qrt_file_read_raw(fn, i_orig); });
    }
}
