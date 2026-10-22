// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// QRT readers on malformed files: counts and offsets that do not fit the file, and truncated files.
// Every reader must throw std::invalid_argument quickly instead of allocating memory for the counts.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_channel.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>

namespace
{
    // A valid 136-byte v5 file: 1 frequency, 1 CIR, 1 RX, 1 TX with one LOS path.
    // Byte offsets: no_orig 11, no_cir 15, no_dest 19, no_freq 23, orig_index 76, path_data_index 91, no_path 99.
    static std::vector<char> qrt_small_file(const std::string &fn)
    {
        quadriga_lib::qrt_file_init<double>(fn, arma::vec{3.0}, arma::mat{{1.0, 2.0, 3.0}}, arma::mat());
        std::vector<quadriga_lib::path> pd(1);
        quadriga_lib::qrt_file_append<double>(fn, pd, arma::vec{0.0, 0.0, 10.0});
        std::ifstream i(fn, std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>());
    }

    static void qrt_write_bytes(const std::string &fn, const std::vector<char> &data, size_t len)
    {
        std::ofstream o(fn, std::ios::binary | std::ios::trunc);
        o.write(data.data(), (std::streamsize)len);
    }

    // Runs f, requires std::invalid_argument and a run time below 0.1 s
    static void qrt_require_fast_throw(const std::function<void()> &f)
    {
        auto t0 = std::chrono::steady_clock::now();
        CHECK_THROWS_AS(f(), std::invalid_argument);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(dt < 0.1);
    }

    static void qrt_parse_all(const std::string &fn)
    {
        arma::uword a, b, c, d;
        arma::uvec co;
        std::vector<std::string> on, dn;
        int v;
        arma::fvec f;
        arma::fmat cp, cr, op, oo;
        quadriga_lib::qrt_file_parse(fn, &a, &b, &c, &d, &co, &on, &dn, &v, &f, &cp, &cr, &op, &oo);
    }

    static void qrt_read_all(const std::string &fn)
    {
        arma::vec fc, tp, to, rp, ro, pl, aod, eod, aoa, eoa;
        arma::mat fbs, lbs, pg;
        arma::cube M;
        std::vector<arma::mat> pc;
        arma::u32_vec ni;
        arma::fmat co;
        std::vector<uint8_t> it;
        quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, &fc, &tp, &to, &rp, &ro, &fbs, &lbs, &pg, &pl, &M,
                                            &aod, &eod, &aoa, &eoa, &pc, 1, &ni, &co, &it);
    }
}

TEST_CASE("QRT Reader - counts and offsets that do not fit the file")
{
    // Patched counts must not make the readers allocate memory or loop over them
    const std::string fn = "tests/data/test_qrt_malformed.qrt";
    const std::vector<char> good = qrt_small_file(fn);
    REQUIRE(good.size() == 136);

    // The unpatched file reads back
    {
        arma::vec pl;
        quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &pl);
        CHECK(pl.n_elem == 1);
        CHECK(quadriga_lib::qrt_file_read_raw(fn, 0).size() == 1);
    }

    struct patch
    {
        const char *name;
        size_t pos;
        unsigned long long value;
        size_t bytes;
        bool header; // false: only the path readers see it
    };
    const std::vector<patch> patches = {
        {"no_orig", 11, 100000000ULL, 4, true},
        {"no_orig max", 11, 0xFFFFFFFFULL, 4, true},
        {"no_cir", 15, 100000000ULL, 4, true},
        {"no_dest", 19, 100000000ULL, 4, true},
        {"no_dest max", 19, 0xFFFFFFFFULL, 4, true},
        {"no_freq", 23, 100000000ULL, 4, true},
        {"orig_index", 76, 1000000000000ULL, 8, true},
        {"no_path", 99, 100000000ULL, 4, false},
        {"no_path max", 99, 0xFFFFFFFFULL, 4, false},
        {"data_offset", 91, 1000000000000ULL, 8, false}};

    for (const auto &p : patches)
    {
        DYNAMIC_SECTION(p.name)
        {
            std::vector<char> bad = good;
            std::memcpy(bad.data() + p.pos, &p.value, p.bytes);
            qrt_write_bytes(fn, bad, bad.size());

            qrt_require_fast_throw([&]
                                   { qrt_read_all(fn); });
            qrt_require_fast_throw([&]
                                   { quadriga_lib::qrt_file_read_raw(fn, 0); });
            if (p.header)
            {
                qrt_require_fast_throw([&]
                                       { qrt_parse_all(fn); });
                qrt_require_fast_throw([&]
                                       { quadriga_lib::qrt_read_cache_init(fn); });
            }
            else // the cache is valid, the path block is not
            {
                std::ifstream f(fn, std::ios::binary);
                auto cache = quadriga_lib::qrt_read_cache_init(fn, &f);
                qrt_require_fast_throw([&]
                                       { arma::vec pl;
                                         quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                             nullptr, nullptr, nullptr, &pl, nullptr, nullptr, nullptr,
                                                                             nullptr, nullptr, nullptr, 1, nullptr, nullptr, nullptr, &f, &cache); });
                f.clear();
                qrt_require_fast_throw([&]
                                       { quadriga_lib::qrt_file_read_raw(fn, 0, &f, &cache); });
            }
        }
    }
    std::remove(fn.c_str());
}

TEST_CASE("QRT Reader - truncated files")
{
    // Every truncation of the small file makes the readers throw std::invalid_argument
    const std::string fn = "tests/data/test_qrt_truncated.qrt";
    const std::vector<char> good = qrt_small_file(fn);
    REQUIRE(good.size() == 136);

    for (size_t len = 0; len < good.size(); ++len)
    {
        qrt_write_bytes(fn, good, len);
        INFO("file length " << len);
        qrt_require_fast_throw([&]
                               { qrt_read_all(fn); });
        qrt_require_fast_throw([&]
                               { quadriga_lib::qrt_file_read_raw(fn, 0); });
        if (len < 99) // header, TX block start and path_data_index
            qrt_require_fast_throw([&]
                                   { quadriga_lib::qrt_read_cache_init(fn); });
        if (len < 87) // header and TX name
            qrt_require_fast_throw([&]
                                   { qrt_parse_all(fn); });
    }
    std::remove(fn.c_str());
}
