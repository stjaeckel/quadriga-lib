// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// New test cases for the QRT write path (qrt_file_init / qrt_file_append / qrt_file_read_raw)
// and reserved-but-unwritten origin reads. Intended to be appended to test_qrt_reader.cpp.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_channel.hpp"

#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>

using quadriga_lib::path;

namespace
{
    // Build one EM path: nSEG segments at deterministic coordinates, identity-plus-offset
    // Jones per frequency, and per-segment type codes. iC selects the target CIR.
    static path make_em_path(unsigned iC, unsigned nSEG, unsigned nFreq,
                             const std::vector<uint8_t> &types)
    {
        path p(nSEG, nFreq, false);
        p.iC = iC;
        for (unsigned s = 0; s < nSEG; ++s)
        {
            float *c = p.coord(s);
            c[0] = (float)(iC * 100 + s);
            c[1] = (float)(s * 2);
            c[2] = (float)(s * 3);
        }
        // Distinct value per (freq, entry) so an interleave bug is visible.
        for (unsigned f = 0; f < nFreq; ++f)
        {
            float *m = p.xpr_coeff(f);
            for (unsigned k = 0; k < 8; ++k)
                m[k] = (float)(f * 10 + k) + 0.5f;
        }
        if (nSEG != 0)
            p.set_interaction_type_codes(types);
        return p;
    }

    static path make_scalar_path(unsigned iC, unsigned nSEG, unsigned nFreq,
                                 const std::vector<uint8_t> &types)
    {
        path p(nSEG, nFreq, true);
        p.iC = iC;
        for (unsigned s = 0; s < nSEG; ++s)
        {
            float *c = p.coord(s);
            c[0] = (float)(iC * 100 + s);
            c[1] = (float)(s * 2);
            c[2] = (float)(s * 3);
        }
        for (unsigned f = 0; f < nFreq; ++f)
        {
            float *m = p.xpr_coeff(f);
            m[0] = (float)(f * 10) + 0.25f;
            m[1] = (float)(f * 10) + 0.75f;
        }
        if (nSEG != 0)
            p.set_interaction_type_codes(types);
        return p;
    }
}

TEST_CASE("QRT Write - init writes a parseable header")
{
    std::string fn = "tests/data/test_write_init.qrt";

    arma::fvec freq = {3.75f};
    arma::fmat cir_pos(4, 3, arma::fill::zeros);
    for (arma::uword i = 0; i < 4; ++i)
        cir_pos(i, 0) = (float)(10 * (i + 1));
    arma::fmat cir_orient; // empty -> cir_fmt 0
    std::vector<std::string> dest_names = {"RX0", "RX1"};
    arma::u32_vec cir_offset = {0u, 2u};

    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, 3, false);

    arma::uword no_cir, no_orig, no_dest, no_freq;
    int version;
    arma::fvec fGHz;
    arma::fmat rd_cir_pos, rd_cir_orient, rd_orig_pos, rd_orig_orient;
    arma::uvec rd_offset;
    std::vector<std::string> orig_names, rd_dest_names;

    quadriga_lib::qrt_file_parse(fn, &no_cir, &no_orig, &no_dest, &no_freq, &rd_offset,
                                 &orig_names, &rd_dest_names, &version, &fGHz,
                                 &rd_cir_pos, &rd_cir_orient, &rd_orig_pos, &rd_orig_orient);

    CHECK(version == 5); // EM
    CHECK(no_orig == 3ULL);
    CHECK(no_cir == 4ULL);
    CHECK(no_dest == 2ULL);
    CHECK(no_freq == 1ULL);
    REQUIRE(fGHz.n_elem == 1);
    CHECK(std::abs(fGHz(0) - 3.75f) < 1e-6f);

    // CIR positions round-trip.
    REQUIRE(rd_cir_pos.n_rows == 4);
    for (arma::uword i = 0; i < 4; ++i)
        CHECK(std::abs(rd_cir_pos(i, 0) - (float)(10 * (i + 1))) < 1e-4f);

    // RX names and offsets round-trip.
    REQUIRE(rd_dest_names.size() == 2);
    CHECK(rd_dest_names[0] == "RX0");
    CHECK(rd_dest_names[1] == "RX1");
    CHECK(rd_offset(1) == 2ULL);

    // All three origin slots are reserved but unwritten -> empty names.
    REQUIRE(orig_names.size() == 3);
    for (const auto &nm : orig_names)
        CHECK(nm.empty());

    std::remove(fn.c_str());
}

TEST_CASE("QRT Write - scalar init selects v6")
{
    std::string fn = "tests/data/test_write_scalar.qrt";

    arma::fvec freq = {125.0f, 250.0f, 500.0f};
    arma::fmat cir_pos(2, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> dest_names = {"RX"};
    arma::u32_vec cir_offset = {0u};

    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, 1, true);

    arma::uword no_freq;
    int version;
    quadriga_lib::qrt_file_parse(fn, nullptr, nullptr, nullptr, &no_freq, nullptr,
                                 nullptr, nullptr, &version);
    CHECK(version == 6);
    CHECK(no_freq == 3ULL);

    std::remove(fn.c_str());
}

TEST_CASE("QRT Write - init argument validation")
{
    std::string fn = "tests/data/test_write_bad.qrt";
    arma::fmat cir_pos(2, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> names = {"RX"};
    arma::u32_vec off = {0u};

    // Empty frequency vector.
    CHECK_THROWS_AS(quadriga_lib::qrt_file_init<float>(fn, arma::fvec(), cir_pos, cir_orient, names, off, 1, false),
                    std::invalid_argument);
    // no_orig == 0.
    CHECK_THROWS_AS(quadriga_lib::qrt_file_init<float>(fn, arma::fvec{1.0f}, cir_pos, cir_orient, names, off, 0, false),
                    std::invalid_argument);
    // cir_pos wrong shape.
    CHECK_THROWS_AS(quadriga_lib::qrt_file_init<float>(fn, arma::fvec{1.0f}, arma::fmat(2, 2), cir_orient, names, off, 1, false),
                    std::invalid_argument);
    // names / offset length mismatch.
    CHECK_THROWS_AS(quadriga_lib::qrt_file_init<float>(fn, arma::fvec{1.0f}, cir_pos, cir_orient,
                                                       std::vector<std::string>{"a", "b"}, off, 1, false),
                    std::invalid_argument);
}

TEST_CASE("QRT Write - init leaves an existing file unchanged when an input is invalid")
{
    // All inputs are checked before the file is opened: a rejected call does not truncate or partly write the file
    std::string fn = "tests/data/test_write_keep.qrt";
    const std::string content = "existing content, not a QRT file";
    {
        std::ofstream f(fn, std::ios::binary | std::ios::trunc);
        f << content;
    }
    auto slurp = [&fn]()
    {
        std::ifstream f(fn, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };
    arma::fmat cir_pos(2, 3, arma::fill::zeros);
    arma::fmat cir_orient;

    // RX name with 256 characters (the second of two names)
    std::vector<std::string> names = {"RX", std::string(256, 'a')};
    arma::u32_vec off = {0u, 1u};
    CHECK_THROWS_AS(quadriga_lib::qrt_file_init<float>(fn, arma::fvec{1.0f}, cir_pos, cir_orient, names, off, 1, false),
                    std::invalid_argument);
    CHECK(slurp() == content);

    // The first CIR offset must be 0 (the reader rejects other files)
    names = {"RX1", "RX2"};
    off = {1u, 1u};
    CHECK_THROWS_AS(quadriga_lib::qrt_file_init<float>(fn, arma::fvec{1.0f}, cir_pos, cir_orient, names, off, 1, false),
                    std::invalid_argument);
    CHECK(slurp() == content);

    // A name with 255 characters is valid and reads back
    names = {std::string(255, 'b')};
    off = {0u};
    quadriga_lib::qrt_file_init<float>(fn, arma::fvec{1.0f}, cir_pos, cir_orient, names, off, 1, false);
    std::vector<std::string> names_read;
    quadriga_lib::qrt_file_parse(fn, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &names_read);
    CHECK(names_read == names);
    std::remove(fn.c_str());
}

TEST_CASE("QRT Write - append then read_raw round-trip (EM)")
{
    std::string fn = "tests/data/test_write_rt.qrt";
    unsigned nFreq = 2, nCir = 3, nOrig = 2;

    arma::fvec freq = {3.75f, 5.0f};
    arma::fmat cir_pos(nCir, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> dest_names = {"RX"};
    arma::u32_vec cir_offset = {0u};

    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, nOrig, false);

    // Build paths for origin 0: CIR 0 gets a LOS + a 1-reflection; CIR 2 gets an 8-segment path
    // that crosses the inline/history boundary. CIR 1 stays empty.
    std::vector<path> paths;
    paths.push_back(make_em_path(0, 0, nFreq, {}));         // LOS
    paths.push_back(make_em_path(0, 1, nFreq, {100}));      // 1 transmission
    std::vector<uint8_t> t8 = {1, 2, 3, 4, 5, 6, 200, 201}; // 6 inline + 2 overflow, last two = reflections
    paths.push_back(make_em_path(2, 8, nFreq, t8));

    arma::fvec tx_pos = {1.0f, 2.0f, 3.0f};
    arma::fvec tx_orient = {0.1f, 0.2f, 0.3f};
    size_t n_written = quadriga_lib::qrt_file_append<float>(fn, paths, tx_pos, tx_orient, "TX0");
    CHECK(n_written == 3);

    // Read origin 0 back.
    std::vector<path> rd = quadriga_lib::qrt_file_read_raw(fn, 0);
    REQUIRE(rd.size() == 3);

    // Group by CIR for order-independent checks (read groups by CIR, generation order within).
    // CIR 0 has the two short paths, CIR 2 has the long one.
    unsigned cir0 = 0, cir2 = 0;
    for (const auto &p : rd)
    {
        if (p.iC == 0)
            ++cir0;
        else if (p.iC == 2)
            ++cir2;
        else
            FAIL("unexpected CIR index " << p.iC);
    }
    CHECK(cir0 == 2);
    CHECK(cir2 == 1);

    // Find the 8-segment path and verify its full round-trip.
    const path *pl = nullptr;
    for (const auto &p : rd)
        if (p.n_seg() == 8)
            pl = &p;
    REQUIRE(pl != nullptr);
    CHECK(pl->iC == 2);
    CHECK(pl->n_freq() == nFreq);
    CHECK_FALSE(pl->is_scalar());

    // Coordinates preserved.
    for (unsigned s = 0; s < 8; ++s)
    {
        const float *c = pl->coord(s);
        CHECK(std::abs(c[0] - (float)(2 * 100 + s)) < 1e-4f);
        CHECK(std::abs(c[1] - (float)(s * 2)) < 1e-4f);
        CHECK(std::abs(c[2] - (float)(s * 3)) < 1e-4f);
    }

    // Type codes preserved across the inline/history boundary.
    std::vector<uint8_t> rt = pl->interaction_type_codes();
    REQUIRE(rt.size() == 8);
    for (unsigned s = 0; s < 8; ++s)
        CHECK(rt[s] == t8[s]);

    // Polarization preserved for every frequency, no interleave swap.
    for (unsigned f = 0; f < nFreq; ++f)
    {
        const float *m = pl->xpr_coeff(f);
        for (unsigned k = 0; k < 8; ++k)
            CHECK(std::abs(m[k] - ((float)(f * 10 + k) + 0.5f)) < 1e-4f);
    }

    std::remove(fn.c_str());
}

TEST_CASE("QRT Write - scalar append then read_raw round-trip (v6)")
{
    std::string fn = "tests/data/test_write_scalar_rt.qrt";
    unsigned nFreq = 5, nCir = 2, nOrig = 1;

    arma::fvec freq = {125.0f, 250.0f, 500.0f, 1000.0f, 2000.0f};
    arma::fmat cir_pos(nCir, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> dest_names = {"RX"};
    arma::u32_vec cir_offset = {0u};

    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, nOrig, true);

    std::vector<path> paths;
    paths.push_back(make_scalar_path(0, 2, nFreq, {50, 60})); // scalar, 2 segments

    arma::fvec tx_pos = {0.0f, 0.0f, 0.0f};
    size_t n = quadriga_lib::qrt_file_append<float>(fn, paths, tx_pos);
    CHECK(n == 1);

    std::vector<path> rd = quadriga_lib::qrt_file_read_raw(fn, 0);
    REQUIRE(rd.size() == 1);
    const path &p = rd[0];
    CHECK(p.is_scalar());
    CHECK(p.n_freq() == nFreq);
    CHECK(p.n_seg() == 2);

    // Scalar coefficients: 2 floats per frequency, correct per-band values.
    for (unsigned f = 0; f < nFreq; ++f)
    {
        const float *m = p.xpr_coeff(f);
        CHECK(std::abs(m[0] - ((float)(f * 10) + 0.25f)) < 1e-4f);
        CHECK(std::abs(m[1] - ((float)(f * 10) + 0.75f)) < 1e-4f);
    }

    std::vector<uint8_t> rt = p.interaction_type_codes();
    REQUIRE(rt.size() == 2);
    CHECK(rt[0] == 50);
    CHECK(rt[1] == 60);

    std::remove(fn.c_str());
}

TEST_CASE("QRT Write - read reserved-but-unwritten origin")
{
    std::string fn = "tests/data/test_write_reserved.qrt";
    unsigned nOrig = 3;

    arma::fvec freq = {3.75f};
    arma::fmat cir_pos(2, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> dest_names = {"RX"};
    arma::u32_vec cir_offset = {0u};

    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, nOrig, false);

    // Write only origin 0 (fills the first free slot); origins 1 and 2 stay reserved.
    std::vector<path> paths = {make_em_path(0, 1, 1, {10})};
    arma::fvec tx_pos = {1.0f, 1.0f, 1.0f};
    quadriga_lib::qrt_file_append<float>(fn, paths, tx_pos, arma::fvec{0.0f, 0.0f, 0.0f}, "TX0");

    // read_raw on the unwritten origin 1 returns empty, does not throw.
    std::vector<path> empty = quadriga_lib::qrt_file_read_raw(fn, 1);
    CHECK(empty.empty());

    // read_raw on written origin 0 returns the one path.
    std::vector<path> got = quadriga_lib::qrt_file_read_raw(fn, 0);
    REQUIRE(got.size() == 1);
    CHECK(got[0].iC == 0);

    // qrt_file_read on the unwritten origin returns empty outputs (no throw).
    arma::Col<double> tx_pos_out, path_length;
    arma::Mat<double> path_gain;
    arma::u32_vec no_int;
    quadriga_lib::qrt_file_read<double>(fn, 0, 1, true, nullptr, &tx_pos_out, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, &path_gain,
                                        &path_length, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, 0, &no_int, nullptr, nullptr);
    CHECK(no_int.n_elem == 0);
    CHECK(path_length.n_elem == 0);

    // Out-of-range origin still throws.
    CHECK_THROWS_AS(quadriga_lib::qrt_file_read_raw(fn, 3), std::out_of_range);

    std::remove(fn.c_str());
}

TEST_CASE("QRT Write - append throws when all slots are full")
{
    std::string fn = "tests/data/test_write_full.qrt";

    arma::fvec freq = {3.75f};
    arma::fmat cir_pos(1, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> dest_names = {"RX"};
    arma::u32_vec cir_offset = {0u};

    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, 2, false);

    std::vector<path> paths = {make_em_path(0, 1, 1, {5})};
    arma::fvec tx_pos = {0.0f, 0.0f, 0.0f};

    CHECK(quadriga_lib::qrt_file_append<float>(fn, paths, tx_pos, arma::fvec{0, 0, 0}, "A") == 1);
    CHECK(quadriga_lib::qrt_file_append<float>(fn, paths, tx_pos, arma::fvec{0, 0, 0}, "B") == 1);
    // Third append exceeds the 2 reserved slots.
    CHECK_THROWS_AS(quadriga_lib::qrt_file_append<float>(fn, paths, tx_pos, arma::fvec{0, 0, 0}, "C"),
                    std::runtime_error);

    std::remove(fn.c_str());
}

TEST_CASE("QRT Write - append validates path against file")
{
    std::string fn = "tests/data/test_write_validate.qrt";

    arma::fvec freq = {3.75f, 5.0f}; // nFreq = 2, EM
    arma::fmat cir_pos(2, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> dest_names = {"RX"};
    arma::u32_vec cir_offset = {0u};

    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, 2, false);
    arma::fvec tx_pos = {0.0f, 0.0f, 0.0f};

    // iC out of range (file has 2 CIRs).
    {
        std::vector<path> bad = {make_em_path(5, 1, 2, {1})};
        CHECK_THROWS_AS(quadriga_lib::qrt_file_append<float>(fn, bad, tx_pos), std::out_of_range);
    }
    // Frequency count mismatch (path has 1 freq, file has 2).
    {
        std::vector<path> bad = {make_em_path(0, 1, 1, {1})};
        CHECK_THROWS_AS(quadriga_lib::qrt_file_append<float>(fn, bad, tx_pos), std::invalid_argument);
    }
    // Layout mode mismatch (scalar path into EM file).
    {
        std::vector<path> bad = {make_scalar_path(0, 1, 2, {1})};
        CHECK_THROWS_AS(quadriga_lib::qrt_file_append<float>(fn, bad, tx_pos), std::invalid_argument);
    }

    std::remove(fn.c_str());
}

TEST_CASE("QRT Write - read_raw with shared stream and cache")
{
    std::string fn = "tests/data/test_write_cache.qrt";
    unsigned nOrig = 2;

    arma::fvec freq = {3.75f};
    arma::fmat cir_pos(2, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> dest_names = {"RX"};
    arma::u32_vec cir_offset = {0u};

    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, nOrig, false);

    std::vector<path> p0 = {make_em_path(0, 2, 1, {1, 2}), make_em_path(1, 1, 1, {3})};
    std::vector<path> p1 = {make_em_path(0, 1, 1, {9})};
    quadriga_lib::qrt_file_append<float>(fn, p0, arma::fvec{0, 0, 0}, arma::fvec{0, 0, 0}, "TX0");
    quadriga_lib::qrt_file_append<float>(fn, p1, arma::fvec{5, 5, 5}, arma::fvec{0, 0, 0}, "TX1");

    std::ifstream stream(fn, std::ios::in | std::ios::binary);
    auto cache = quadriga_lib::qrt_read_cache_init(fn, &stream);

    std::vector<path> r0 = quadriga_lib::qrt_file_read_raw(fn, 0, &stream, &cache);
    std::vector<path> r1 = quadriga_lib::qrt_file_read_raw(fn, 1, &stream, &cache);
    stream.close();

    CHECK(r0.size() == 2);
    CHECK(r1.size() == 1);
    CHECK(r1[0].iC == 0);
    std::vector<uint8_t> t = r1[0].interaction_type_codes();
    REQUIRE(t.size() == 1);
    CHECK(t[0] == 9);

    std::remove(fn.c_str());
}
TEST_CASE("QRT Write - append after an orientation that rounds to 0 in float")
{
    // A CIR orientation column of 1e-50 rad is non-zero in double but 0 in float. Init and append
    // must agree on the stored orientation columns, otherwise the TX table and path data are misplaced.
    for (bool scalar : {false, true})
    {
        std::string fn = scalar ? "tests/data/test_write_tiny_ori_v6.qrt" : "tests/data/test_write_tiny_ori_v5.qrt";
        arma::vec freq = scalar ? arma::vec{1000.0, 2000.0} : arma::vec{3.5, 28.0};
        arma::mat cir_pos = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        arma::mat cir_ori(3, 3, arma::fill::zeros);
        cir_ori.col(0).fill(1e-50); // bank: non-zero in double, 0.0f in float
        cir_ori(1, 2) = 0.25;       // heading of CIR 1 is stored
        quadriga_lib::qrt_file_init<double>(fn, freq, cir_pos, cir_ori, {"RX"}, {0}, 3, scalar);

        auto mk = [&](unsigned iC, unsigned nSEG, std::vector<uint8_t> t)
        { return scalar ? make_scalar_path(iC, nSEG, 2, t) : make_em_path(iC, nSEG, 2, t); };
        std::vector<std::vector<path>> tx(3);
        tx[0] = {mk(0, 0, {}), mk(2, 3, {1, 2, 3}), mk(0, 1, {4})};
        tx[2] = {mk(1, 2, {5, 6})}; // tx[1] has no paths
        for (unsigned t = 0; t < 3; ++t)
            CHECK(quadriga_lib::qrt_file_append<double>(fn, tx[t], arma::vec{100.0 + t, 200.0, 30.0},
                                                        arma::vec{0.0, 0.0, 0.5 * t}, "TX" + std::to_string(t)) == tx[t].size());

        auto cache = quadriga_lib::qrt_read_cache_init(fn);
        REQUIRE(cache.no_orig == 3);
        for (unsigned t = 0; t < 3; ++t)
        {
            CHECK(cache.orig_pos_all(t, 0) == float(100.0 + t));
            CHECK(cache.orig_pos_all(t, 1) == 200.0f);
            CHECK(cache.orig_pos_all(t, 2) == 30.0f);
            CHECK(cache.orig_orientation(t, 2) == float(0.5 * t));
        }
        CHECK(cache.cir_orientation(0, 0) == 0.0f);
        CHECK(cache.cir_orientation(1, 2) == 0.25f);

        // read_raw returns the paths of each TX grouped by CIR, in written order within a CIR
        for (unsigned t = 0; t < 3; ++t)
        {
            std::vector<path> raw = quadriga_lib::qrt_file_read_raw(fn, t);
            std::vector<const path *> exp;
            for (unsigned c = 0; c < 3; ++c)
                for (auto &p : tx[t])
                    if (p.iC == c)
                        exp.push_back(&p);
            REQUIRE(raw.size() == exp.size());
            for (size_t i = 0; i < raw.size(); ++i)
            {
                CHECK(raw[i].iC == exp[i]->iC);
                REQUIRE(raw[i].n_seg() == exp[i]->n_seg());
                CHECK(raw[i].interaction_type_codes() == exp[i]->interaction_type_codes());
                for (size_t s = 0; s < raw[i].n_seg(); ++s)
                    for (int k = 0; k < 3; ++k)
                        CHECK(raw[i].coord(s)[k] == exp[i]->coord(s)[k]);
                for (size_t f = 0; f < 2; ++f)
                    for (size_t k = 0; k < (scalar ? 2u : 8u); ++k)
                        CHECK(raw[i].xpr_coeff(f)[k] == exp[i]->xpr_coeff(f)[k]);
            }
        }

        // qrt_file_read returns the CIR positions and the path count of each CIR
        for (unsigned c = 0; c < 3; ++c)
        {
            arma::vec rx_pos, rx_ori;
            arma::u32_vec no_int;
            quadriga_lib::qrt_file_read<double>(fn, c, 0, true, nullptr, nullptr, nullptr, &rx_pos, &rx_ori, nullptr, nullptr,
                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, &no_int);
            CHECK(arma::approx_equal(rx_pos, cir_pos.row(c).t(), "absdiff", 0.0));
            CHECK(rx_ori(2) == (c == 1 ? 0.25 : 0.0));
            CHECK(no_int.n_elem == (c == 0 ? 2u : c == 2 ? 1u : 0u));
        }
        std::remove(fn.c_str());
    }
}

TEST_CASE("QRT Write - append to a v4 file is rejected")
{
    // v4 legacy files cannot be appended to; the file must stay unchanged
    std::string src = "tests/data/test.qrt", fn = "tests/data/test_write_v4_copy.qrt";
    std::vector<char> before;
    {
        std::ifstream i(src, std::ios::binary);
        before.assign(std::istreambuf_iterator<char>(i), std::istreambuf_iterator<char>());
        std::ofstream o(fn, std::ios::binary);
        o.write(before.data(), (std::streamsize)before.size());
    }
    std::vector<path> paths = {make_em_path(0, 1, 1, {1})};
    CHECK_THROWS_AS(quadriga_lib::qrt_file_append<double>(fn, paths, arma::vec{0.0, 0.0, 0.0}), std::invalid_argument);
    std::ifstream i(fn, std::ios::binary);
    std::vector<char> after((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>());
    CHECK(after == before);
    i.close();
    std::remove(fn.c_str());
}
