// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// QRT readers on malformed files: counts and offsets that do not fit the file, and truncated files.
// Every reader must throw std::runtime_error quickly instead of allocating memory for the counts.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_channel.hpp"
#include "qd_helpers_wrappers.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>

namespace
{
    // A valid 136-byte v5 file: 1 frequency, 1 CIR, 1 RX, 1 TX with one LOS path.
    // Byte offsets: n_orig 11, n_cir 15, n_dest 19, n_freq 23, orig_index 76, path_data_index 91, n_path 99.
    static std::vector<char> qrt_small_file(const std::string &fn)
    {
        quadriga_lib::qrt_file_init<double>(fn, arma::vec{3.0e9}, arma::mat{{1.0, 2.0, 3.0}}, arma::mat());
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

    // Runs f, requires std::runtime_error and a run time below 1 s (a throw takes under 1 ms; an allocation from the
    // counts in the file would take seconds)
    static void qrt_require_fast_throw(const std::function<void()> &f)
    {
        auto t0 = std::chrono::steady_clock::now();
        CHECK_THROWS_AS(f(), std::runtime_error);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(dt < 1.0);
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
        arma::vec fc;
        arma::mat tp, to, rp, ro;
        std::vector<arma::vec> pl, aod, eod, aoa, eoa;
        std::vector<arma::mat> fbs, lbs, pg;
        std::vector<arma::cube> M;
        std::vector<std::vector<arma::mat>> pc;
        std::vector<arma::u32_vec> ni;
        std::vector<arma::fmat> co;
        std::vector<std::vector<uint8_t>> it;
        quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, &fc, &tp, &to, &rp, &ro, &fbs, &lbs, &pg, &pl, &M,
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
        std::vector<arma::vec> pl;
        quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &pl);
        CHECK(pl.at(0).n_elem == 1);
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
        {"n_orig", 11, 100000000ULL, 4, true},
        {"n_orig max", 11, 0xFFFFFFFFULL, 4, true},
        {"n_cir", 15, 100000000ULL, 4, true},
        {"n_dest", 19, 100000000ULL, 4, true},
        {"n_dest max", 19, 0xFFFFFFFFULL, 4, true},
        {"n_freq", 23, 100000000ULL, 4, true},
        {"orig_index", 76, 1000000000000ULL, 8, true},
        {"n_path", 99, 100000000ULL, 4, false},
        {"n_path max", 99, 0xFFFFFFFFULL, 4, false},
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
                                       { std::vector<arma::vec> pl;
                                         quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr,
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
    // Every truncation of the small file makes the readers throw std::runtime_error
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

TEST_CASE("QRT Reader - v4 file without origins")
{
    // A v4 file stores its frequency in the first origin block. Without origins, the cache holds one
    // frequency of 0 Hz, as qrt_file_read does for an unfilled v4 slot, so that freq.n_elem == n_freq.
    const std::string fn = "tests/data/test_qrt_v4_no_orig.qrt";
    std::vector<char> data(23, 0); // "#QRT-BINv04", then n_orig = n_cir = n_dest = 0
    std::memcpy(data.data(), "#QRT-BINv04", 11);
    qrt_write_bytes(fn, data, data.size());

    auto cache = quadriga_lib::qrt_read_cache_init(fn);
    CHECK(cache.version == 4);
    CHECK(cache.n_orig == 0);
    CHECK(cache.n_freq == 1);
    REQUIRE(cache.freq.n_elem == cache.n_freq);
    CHECK(cache.freq(0) == 0.0f);

    std::remove(fn.c_str());
}

TEST_CASE("QRT Reader - CIRs that share one path block")
{
    // Path blocks of different CIRs must not overlap; otherwise a small file could return its largest block once per
    // CIR, and the output would grow with the square of the file size
    const std::string fn = "tests/data/test_qrt_shared_block.qrt";
    const arma::uword n_cir = 4;
    quadriga_lib::qrt_file_init<double>(fn, arma::vec{3.0e9}, arma::mat(n_cir, 3, arma::fill::ones), arma::mat(), {"RX"},
                                        {0}, 2);
    std::vector<quadriga_lib::path> pd(50); // all in CIR 0; the other CIRs have no paths
    quadriga_lib::qrt_file_append<double>(fn, pd, arma::vec{0.0, 0.0, 10.0});
    quadriga_lib::qrt_file_append<double>(fn, std::vector<quadriga_lib::path>(1), arma::vec{0.0, 0.0, 10.0}); // origin 1

    std::vector<arma::vec> pl;
    auto read = [&](const arma::uvec &i_cir, arma::uword i_orig = 0)
    {
        quadriga_lib::qrt_file_read<double>(fn, i_cir, i_orig, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, &pl);
    };

    // The written file reads back, also with a CIR listed twice
    read({});
    REQUIRE(pl.size() == n_cir);
    CHECK(pl[0].n_elem == 50);
    CHECK(quadriga_lib::qrt_file_read_raw(fn, 0).size() == 50);
    read({0, 0, 0});
    CHECK(pl.size() == 3);

    std::ifstream i(fn, std::ios::binary);
    const std::vector<char> good((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>());
    i.close();
    const auto cache = quadriga_lib::qrt_read_cache_init(fn);
    const size_t pdo = (size_t)cache.path_data_offset(0), pdo1 = (size_t)cache.path_data_offset(1);
    auto write_index = [&](size_t pos, const std::vector<size_t> &value) // writes 'value' at byte 'pos' of the good file
    {
        std::vector<char> data = good;
        std::memcpy(data.data() + pos, value.data(), value.size() * sizeof(size_t));
        qrt_write_bytes(fn, data, data.size());
    };
    std::vector<size_t> idx(n_cir); // path_data_index of origin 0
    std::memcpy(idx.data(), good.data() + pdo, n_cir * sizeof(size_t));

    // All CIRs point to the block of CIR 0: each block runs into the next one
    write_index(pdo, {idx[0], idx[0], idx[0], idx[0]});
    qrt_require_fast_throw([&]
                           { read({}); });
    qrt_require_fast_throw([&]
                           { read({1}); });
    qrt_require_fast_throw([&]
                           { quadriga_lib::qrt_file_read_raw(fn, 0); });

    // CIRs 0 and 2 point to the block of CIR 0, each followed by a valid offset: a list with gaps reads it twice, more
    // than the file holds, also when the wrappers read the list in chunks of one CIR
    write_index(pdo, {idx[0], idx[1], idx[0], idx[3]});
    read({0});
    CHECK(pl.at(0).n_elem == 50);
    read({2, 2}); // one CIR listed twice
    CHECK(pl.size() == 2);
    qrt_require_fast_throw([&]
                           { read({0, 2}); });
    struct no_unlock
    {
    };
    auto read_chunked = [&](const arma::uvec &i_cir)
    {
        qd_qrt_read_chunked<no_unlock>(fn, i_cir, 0, false, 1, 9, [](const arma::vec &, arma::uword) {},
                                       [](const qd_qrt_chunk &, arma::uword) {}, 1);
    };
    read_chunked({0, 0}); // one CIR listed twice, in two chunks
    qrt_require_fast_throw([&]
                           { read_chunked({0, 2}); });

    // The last CIR of origin 0 points to the path block of origin 1, behind the end of its own origin block
    size_t idx1 = 0; // first entry of the path_data_index of origin 1
    std::memcpy(&idx1, good.data() + pdo1, sizeof(size_t));
    write_index(pdo + (n_cir - 1) * sizeof(size_t), {idx1});
    qrt_require_fast_throw([&]
                           { read({n_cir - 1}); });

    // CIR 0 of origin 1 points to the path block of origin 0, before its own path_data_index
    write_index(pdo1, {idx[0]});
    qrt_require_fast_throw([&]
                           { read({0}, 1); });
    qrt_require_fast_throw([&]
                           { quadriga_lib::qrt_file_read_raw(fn, 1); });

    // Origin 1 uses the data block of origin 0
    {
        std::vector<char> data = good;
        size_t pos = 0; // orig_index entry of origin 1: the 8 bytes after the entry of origin 0 in the header
        for (size_t p = 0; p + 16 <= pdo; ++p)
            if (std::memcmp(data.data() + p, &cache.orig_index[0], 8) == 0 && std::memcmp(data.data() + p + 8, &cache.orig_index[1], 8) == 0)
                pos = p + 8;
        REQUIRE(pos != 0);
        std::memcpy(data.data() + pos, &cache.orig_index[0], 8);
        qrt_write_bytes(fn, data, data.size());
    }
    qrt_require_fast_throw([&]
                           { quadriga_lib::qrt_read_cache_init(fn); });

    std::remove(fn.c_str());
}

TEST_CASE("QRT Reader - destination offsets that decrease or point beyond the end of the CIRs")
{
    // A file with 3 CIRs and 3 destinations at the CIR offsets {0, 1, 2}; patched offsets must throw
    const std::string fn = "tests/data/test_qrt_malformed_dest.qrt";
    quadriga_lib::qrt_file_init<double>(fn, arma::vec{3.0e9}, arma::mat(3, 3, arma::fill::zeros), arma::mat(),
                                        {"a", "b", "c"}, arma::u32_vec{0u, 1u, 2u});
    std::vector<char> good;
    {
        std::ifstream i(fn, std::ios::binary);
        good.assign(std::istreambuf_iterator<char>(i), std::istreambuf_iterator<char>());
    }
    const char pattern[] = {0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 'a'};
    auto it = std::search(good.begin(), good.end(), std::begin(pattern), std::end(pattern));
    REQUIRE(it != good.end());
    const size_t at = (size_t)(it - good.begin());
    REQUIRE_NOTHROW(qrt_parse_all(fn));
    {
        std::vector<char> empty_last = good; // {0, 1, 3}: the last destination has no CIRs
        empty_last[at + 8] = 3;
        qrt_write_bytes(fn, empty_last, empty_last.size());
        CHECK_NOTHROW(qrt_parse_all(fn));
    }
    for (const auto &patch : std::vector<std::pair<size_t, char>>{{8, 0}, {8, 4}}) // {0, 1, 0} and {0, 1, 4}
    {
        std::vector<char> bad = good;
        bad[at + patch.first] = patch.second;
        qrt_write_bytes(fn, bad, bad.size());
        qrt_require_fast_throw([&] { qrt_parse_all(fn); });
    }
    std::remove(fn.c_str());
}
