// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// QRT ray-tracing files: qrt_file_parse, qrt_read_cache_init, then qrt_file_read with all outputs (up to two origins;
// CIR lists of one CIR each, one index past the end, two CIRs with a cache, all CIRs; both directions, both
// normalizations of M) and qrt_file_read_raw

#include "fuzz_target.hpp"

// One CIR with a LOS path and a path with two interactions, EM (8 Jones values) or scalar (2 values) per frequency
static std::vector<quadriga_lib::path> two_paths(unsigned i_cir, unsigned n_freq, bool scalar)
{
    std::vector<quadriga_lib::path> paths;
    for (unsigned n_seg : {0u, 2u})
    {
        quadriga_lib::path p(n_seg, n_freq, scalar);
        p.i_cir = i_cir;
        for (unsigned s = 0; s < n_seg; ++s)
        {
            float *c = p.coord(s);
            c[0] = 10.0f + (float)s, c[1] = 2.0f * (float)s, c[2] = 1.5f;
        }
        for (unsigned f = 0; f < n_freq; ++f)
        {
            float *m = p.polmat(f);
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
        arma::fvec freq = {3.5e9f, 28.0e9f};
        arma::fmat cir_pos = {{5.0f, 0.0f, 1.5f}, {6.0f, 1.0f, 1.5f}};
        arma::fmat cir_orientation = {{0.0f, 0.0f, 0.1f}, {0.0f, 0.0f, 0.2f}};
        quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orientation, {"RX0", "RX1"}, {0, 1}, 2, scalar);
        for (unsigned i_orig = 0; i_orig < 2; ++i_orig)
        {
            std::vector<quadriga_lib::path> paths = two_paths(0, 2, scalar), more = two_paths(1, 2, scalar);
            paths.insert(paths.end(), more.begin(), more.end());
            quadriga_lib::qrt_file_append<float>(fn, paths, {0.0f, (float)i_orig, 10.0f}, {},
                                                 "TX" + std::to_string(i_orig));
        }
    }
}

void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn = fz::write_file("in.qrt", data, size);

    arma::uword n_cir = 0, n_orig = 0;
    fz::clean([&]
              {
        arma::uword n_dest, n_freq;
        arma::uvec cir_offset;
        std::vector<std::string> orig_names, dest_names;
        int version;
        arma::fvec freq;
        arma::fmat cir_pos, cir_orientation, orig_pos, orig_orientation;
        quadriga_lib::qrt_file_parse(fn, &n_cir, &n_orig, &n_dest, &n_freq, &cir_offset, &orig_names, &dest_names,
                                     &version, &freq, &cir_pos, &cir_orientation, &orig_pos, &orig_orientation); });
    fz::clean([&] { quadriga_lib::qrt_read_cache_init(fn); });

    arma::uword n_orig_try = std::min(n_orig, (arma::uword)2), n_cir_try = std::min(n_cir, (arma::uword)2);
    for (arma::uword i_orig = 0; i_orig <= n_orig_try; ++i_orig) // i_orig == n_orig_try: past the end if the file has one or two
    {
        // Lists: each CIR alone, one index past the end, the first two together (with and without a cache), all CIRs
        std::vector<arma::uvec> lists;
        for (arma::uword i_cir = 0; i_cir <= n_cir_try; ++i_cir)
            lists.push_back({i_cir});
        lists.push_back(arma::uvec{1, 0});
        if (n_cir <= 8) // All CIRs; crafted CIRs can all point to the same path block (see qrt_file_read_raw)
            lists.push_back(arma::uvec());
        for (size_t il = 0; il < lists.size(); ++il)
            for (bool uplink : {false, true})
                fz::clean([&]
                          {
                    arma::vec center_freq;
                    arma::mat tx_pos, tx_orientation, rx_pos, rx_orientation;
                    std::vector<arma::vec> path_length, aod, eod, aoa, eoa;
                    std::vector<arma::mat> fbs_pos, lbs_pos, path_gain;
                    std::vector<arma::cube> M;
                    std::vector<std::vector<arma::mat>> path_coord;
                    std::vector<arma::u32_vec> n_interact;
                    std::vector<arma::fmat> coord;
                    std::vector<std::vector<uint8_t>> interact_type;
                    const arma::uvec &i_cir = lists[il];
                    if (il == (arma::uword)n_cir_try + 1) // {1, 0} with a cache
                    {
                        std::ifstream f(fn, std::ios::binary);
                        auto cache = quadriga_lib::qrt_read_cache_init(fn, &f);
                        quadriga_lib::qrt_file_read<double>(fn, i_cir, i_orig, uplink, &center_freq, &tx_pos,
                                                            &tx_orientation, &rx_pos, &rx_orientation, &fbs_pos, &lbs_pos,
                                                            &path_gain, &path_length, &M, &aod, &eod, &aoa, &eoa,
                                                            &path_coord, uplink ? 0 : 1, &n_interact, &coord, &interact_type,
                                                            &f, &cache);
                    }
                    else
                        quadriga_lib::qrt_file_read<double>(fn, i_cir, i_orig, uplink, &center_freq, &tx_pos,
                                                            &tx_orientation, &rx_pos, &rx_orientation, &fbs_pos, &lbs_pos,
                                                            &path_gain, &path_length, &M, &aod, &eod, &aoa, &eoa,
                                                            &path_coord, uplink ? 0 : 1, &n_interact, &coord, &interact_type); });
        fz::clean([&] { quadriga_lib::qrt_file_read_raw(fn, i_orig); });
    }
}
