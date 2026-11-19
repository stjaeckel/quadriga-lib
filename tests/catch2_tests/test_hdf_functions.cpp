// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Behavior of the public HDF5 channel-file functions, observed through the API only: file creation, storage layout
// and slot indexing, round trips of the structured channel data and of unstructured datasets of every supported type
// and rank, slot overwrite, and the error types. The on-disk tree is checked object by object with the HDF5 C library
// in test_hdf5_interop.cpp; malformed files and files changed by other tools are rejected, which is covered in
// test_hdf_robustness_internal.cpp.
//
// Oracles: the written channel objects themselves. Structured data is stored in single precision, so it comes back
// as the written value rounded to float and converted to the read type, (R)(float)x, which is compared exactly.
// Unstructured data comes back bit-identical with its type; an arma::Row comes back as an arma::Col.
//
// Fixtures: channels with 2 RX and 3 TX antennas whose values differ in every element and are not representable in
// single precision (so the rounding is visible). File names are unique per test case; each file is removed when
// the case starts and when it ends.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_any_type_id.hpp"

#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace
{
    // Removes the file when the test case starts and when it ends (also when a REQUIRE fails)
    struct test_file
    {
        const std::string fn;
        explicit test_file(std::string name) : fn(std::move(name)) { std::remove(fn.c_str()); }
        ~test_file() { std::remove(fn.c_str()); }
        test_file(const test_file &) = delete;
        test_file &operator=(const test_file &) = delete;
    };

    bool file_exists(const std::string &fn)
    {
        return std::ifstream(fn).good();
    }

    // Channel IDs of all slots (0 = empty slot)
    arma::u32_vec channel_ids(const std::string &fn)
    {
        arma::u32_vec id;
        quadriga_lib::hdf5_read_layout(fn, &id);
        return id;
    }

    // Fills an Armadillo object with distinct values start + 0.37, start + 0.74, ... (0.37 is not a float)
    template <typename A>
    A filled(A x, double start, double scale = 1.0)
    {
        x.imbue([&]()
                { return (typename A::elem_type)((start += 0.37) * scale); });
        return x;
    }

    // Fills an integer Armadillo object with 0, step, 2 step, ...
    template <typename A>
    A counted(A x, typename A::elem_type step)
    {
        typename A::elem_type value = 0;
        x.imbue([&]()
                { return (value += step) - step; });
        return x;
    }

    // A channel with every structured field: n_rx = 2, n_tx = 3, one entry of 'n_path' per snapshot. Odd snapshots
    // have delays of size [1, 1, n_path], even ones [n_rx, n_tx, n_path]. With 'per_snapshot', positions,
    // orientations and the center frequency have one column per snapshot, otherwise one column.
    template <typename dtype>
    quadriga_lib::channel<dtype> full_channel(const std::vector<arma::uword> &n_path, bool per_snapshot = true)
    {
        using cube = arma::Cube<dtype>;
        using mat = arma::Mat<dtype>;
        using col = arma::Col<dtype>;
        const arma::uword n_snap = n_path.size(), n_pos = per_snapshot ? n_snap : 1;

        quadriga_lib::channel<dtype> c;
        c.name = "full channel";
        c.initial_position = 2;
        c.center_freq = filled(col(n_pos), 1.0, 1e9);
        c.tx_pos = filled(mat(3, n_pos), 1.0);
        c.rx_pos = filled(mat(3, n_pos), 10.0);
        c.tx_orientation = filled(mat(3, n_pos), -1.0);
        c.rx_orientation = filled(mat(3, n_pos), -2.0);
        for (arma::uword s = 0; s < n_snap; ++s)
        {
            const arma::uword np = n_path[s];
            const double o = 100.0 * (double)s;
            c.coeff_re.push_back(filled(cube(2, 3, np), o));
            c.coeff_im.push_back(filled(cube(2, 3, np), o + 10.0));
            c.delay.push_back(filled(s % 2 ? cube(1, 1, np) : cube(2, 3, np), o + 20.0, 1e-9));
            c.path_gain.push_back(filled(col(np), o + 30.0));
            c.path_length.push_back(filled(col(np), o + 40.0));
            c.polmat.push_back(filled(mat(8, np), o + 50.0));
            c.path_angles.push_back(filled(mat(np, 4), o + 60.0));
            c.fbs_pos.push_back(filled(mat(3, np), o + 70.0));
            c.lbs_pos.push_back(filled(mat(3, np), o + 80.0));
            arma::u32_vec n_interact(np);
            for (arma::uword p = 0; p < np; ++p)
                n_interact(p) = (unsigned)(p % 3); // 0 = LOS, 1, 2
            c.n_interact.push_back(n_interact);
            c.interact_coord.push_back(filled(mat(3, arma::accu(n_interact)), o + 90.0));
        }
        return c;
    }

    // A small valid channel: 1 snapshot, 2 RX, 3 TX, 'n_path' paths
    quadriga_lib::channel<double> small_channel(const std::string &name, arma::uword n_path = 2)
    {
        quadriga_lib::channel<double> c;
        c.name = name;
        c.tx_pos = filled(arma::mat(3, 1), 1.0);
        c.rx_pos = filled(arma::mat(3, 1), 5.0);
        c.coeff_re.push_back(filled(arma::cube(2, 3, n_path), 0.0));
        c.coeff_im.push_back(filled(arma::cube(2, 3, n_path), 1.0));
        c.delay.push_back(filled(arma::cube(2, 3, n_path), 2.0, 1e-9));
        return c;
    }

    // Read-back oracle of structured data: the value stored in single precision, converted to the read type R
    template <typename R, template <typename> class A, typename W>
    A<R> stored(const A<W> &x)
    {
        return arma::conv_to<A<R>>::from(arma::conv_to<A<float>>::from(x));
    }

    // Every structured field of the read channel 'r' equals the single-precision value of the written channel 'c'
    template <typename R, typename W>
    void check_structured(const quadriga_lib::channel<R> &r, const quadriga_lib::channel<W> &c)
    {
        auto same = [](const auto &a, const auto &b)
        { return arma::size(a) == arma::size(b) && arma::approx_equal(a, b, "absdiff", 0.0); };
        auto same_list = [&](const auto &read, const auto &written, const char *field)
        {
            INFO("field " << field);
            REQUIRE(read.size() == written.size());
            for (size_t s = 0; s < read.size(); ++s)
            {
                INFO("snapshot " << s);
                CHECK(same(read[s], stored<R>(written[s])));
            }
        };

        CHECK(r.is_valid().empty());
        CHECK(r.name == c.name);
        CHECK(r.n_snap() == c.n_snap());
        CHECK(r.initial_position == c.initial_position);
        CHECK(same(r.center_freq, stored<R>(c.center_freq)));
        CHECK(same(r.tx_pos, stored<R>(c.tx_pos)));
        CHECK(same(r.rx_pos, stored<R>(c.rx_pos)));

        // Empty orientations come back as [3, 1] zeros (if the channel has snapshots)
        const arma::Mat<R> zeros = c.n_snap() == 0 ? arma::Mat<R>() : arma::Mat<R>(3, 1, arma::fill::zeros);
        CHECK(same(r.tx_orientation, c.tx_orientation.n_elem ? stored<R>(c.tx_orientation) : zeros));
        CHECK(same(r.rx_orientation, c.rx_orientation.n_elem ? stored<R>(c.rx_orientation) : zeros));

        same_list(r.coeff_re, c.coeff_re, "coeff_re");
        same_list(r.coeff_im, c.coeff_im, "coeff_im");
        same_list(r.delay, c.delay, "delay");
        same_list(r.path_gain, c.path_gain, "path_gain");
        same_list(r.path_length, c.path_length, "path_length");
        same_list(r.polmat, c.polmat, "polmat");
        same_list(r.path_angles, c.path_angles, "path_angles");
        same_list(r.fbs_pos, c.fbs_pos, "fbs_pos");
        same_list(r.lbs_pos, c.lbs_pos, "lbs_pos");
        same_list(r.interact_coord, c.interact_coord, "interact_coord");
        REQUIRE(r.n_interact.size() == c.n_interact.size());
        for (size_t s = 0; s < c.n_interact.size(); ++s)
            CHECK(same(r.n_interact[s], c.n_interact[s]));
    }

    // Writes 'c' and checks the read-back as float and as double
    template <typename W>
    void check_round_trip(const std::string &fn, const quadriga_lib::channel<W> &c, unsigned ix = 0)
    {
        quadriga_lib::hdf5_write_channel(fn, c, ix);
        check_structured(quadriga_lib::hdf5_read_channel<float>(fn, ix), c);
        check_structured(quadriga_lib::hdf5_read_channel<double>(fn, ix), c);
    }

    // True if 'b' holds the same value as 'a' with the same type; an arma::Row in 'a' must come back as arma::Col
    bool same_any(const std::any &a, const std::any &b)
    {
        unsigned long long da[3], db[3];
        const void *pa = nullptr, *pb = nullptr;
        int ta = qd_any_type_id(&a, da, &pa), tb = qd_any_type_id(&b, db, &pb);
        if (ta >= 50)
            ta -= 10;
        if (ta < 0 || ta != tb)
            return false;
        if (ta == 9)
            return std::any_cast<std::string>(a) == std::any_cast<std::string>(b);
        if (da[0] != db[0] || da[1] != db[1] || da[2] != db[2])
            return false;
        const size_t elem_size = (ta % 10 == 0 || ta % 10 >= 4) ? 4 : 8; // float, uint32, int32 : 64-bit types
        const size_t n_bytes = (size_t)(da[0] * da[1] * da[2]) * elem_size;
        return n_bytes == 0 || std::memcmp(pa, pb, n_bytes) == 0; // Empty objects may have null pointers
    }

    // One unstructured dataset of every type ID supported by the HDF5 functions (any_type_id 9 to 55)
    void all_par_types(std::vector<std::string> &names, std::vector<std::any> &data)
    {
        auto add = [&](const char *name, std::any value)
        { names.push_back(name), data.push_back(std::move(value)); };
        add("string", std::string("Bla Bla Bla"));
        add("scalar_float", 3.14f);
        add("scalar_double", 6.28);
        add("scalar_ullong", 12341100000ULL);
        add("scalar_llong", -98760000000LL);
        add("scalar_unsigned", 13U);
        add("scalar_int", -13);
        add("mat_float", filled(arma::fmat(3, 6), 0.0));
        add("mat_double", filled(arma::mat(4, 6), 1.0));
        add("mat_uword", counted(arma::umat(5, 6), 12341100000ULL));
        add("mat_sword", counted(arma::imat(5, 6), -98760000000LL));
        add("mat_unsigned", counted(arma::Mat<unsigned>(4, 3), 1U));
        add("mat_int", counted(arma::Mat<int>(4, 3), -1));
        add("cube_float", filled(arma::fcube(3, 6, 2), 2.0));
        add("cube_double", filled(arma::cube(4, 6, 3), 3.0));
        add("cube_uword", counted(arma::ucube(5, 6, 2), 12341100000ULL));
        add("cube_sword", counted(arma::icube(5, 6, 3), -98760000000LL));
        add("cube_unsigned", counted(arma::Cube<unsigned>(4, 3, 2), 1U));
        add("cube_int", counted(arma::Cube<int>(4, 3, 5), -1));
        add("col_float", filled(arma::fvec(13), 4.0));
        add("col_double", filled(arma::vec(11), 5.0));
        add("col_uword", counted(arma::uvec(8), 221ULL));
        add("col_sword", counted(arma::ivec(18), -13LL));
        add("col_unsigned", counted(arma::Col<unsigned>(13), 11U));
        add("col_int", counted(arma::Col<int>(13), -13));
        add("row_float", filled(arma::frowvec(12), 6.0));
        add("row_double", filled(arma::rowvec(10), 7.0));
        add("row_uword", counted(arma::urowvec(7), 111ULL));
        add("row_sword", counted(arma::irowvec(8), -17LL));
        add("row_unsigned", counted(arma::Row<unsigned>(14), 3U));
        add("row_int", counted(arma::Row<int>(12), -4));
    }
}

// ---- hdf5_create_file / hdf5_read_layout / hdf5_reshape_layout ----

TEST_CASE("hdf5_create_file - creates a file with the given layout and all slots empty")
{
    test_file f("test_hdf5_create.hdf5");

    quadriga_lib::hdf5_create_file(f.fn, 2, 3, 4, 5);
    arma::u32_vec id = {7, 7}; // Stale content is replaced
    arma::u32_vec layout = quadriga_lib::hdf5_read_layout(f.fn, &id);
    REQUIRE(layout.n_elem == 4);
    CHECK(layout(0) == 2);
    CHECK(layout(1) == 3);
    CHECK(layout(2) == 4);
    CHECK(layout(3) == 5);
    CHECK(id.n_elem == 120);
    CHECK(arma::all(id == 0));

    // Default layout 65536 x 1 x 1 x 1
    std::remove(f.fn.c_str());
    quadriga_lib::hdf5_create_file(f.fn);
    layout = quadriga_lib::hdf5_read_layout(f.fn, &id);
    CHECK(layout(0) == 65536);
    CHECK(layout(1) == 1);
    CHECK(layout(2) == 1);
    CHECK(layout(3) == 1);
    CHECK(id.n_elem == 65536);

    // An existing file is refused and kept; after removing it, a new file has all slots empty
    auto c = small_channel("old");
    quadriga_lib::hdf5_write_channel(f.fn, c, 1);
    CHECK_THROWS_AS(quadriga_lib::hdf5_create_file(f.fn, 3), std::runtime_error);
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 1).name == "old");
    std::remove(f.fn.c_str());
    quadriga_lib::hdf5_create_file(f.fn, 3);
    id = channel_ids(f.fn);
    CHECK(id.n_elem == 3);
    CHECK(arma::all(id == 0));
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 1).empty());
}

TEST_CASE("hdf5_create_file - input validation")
{
    test_file f("test_hdf5_create_validation.hdf5");

    CHECK_NOTHROW(quadriga_lib::hdf5_create_file(f.fn, 2, 2, 2, 2));
    std::remove(f.fn.c_str());

    SECTION("zero dimension")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_create_file(f.fn, 0, 2, 2, 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_create_file(f.fn, 2, 0, 2, 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_create_file(f.fn, 2, 2, 0, 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_create_file(f.fn, 2, 2, 2, 0), std::invalid_argument);
        CHECK(!file_exists(f.fn));
    }
    SECTION("more than 2^32 - 1 slots, rejected before the file is created")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_create_file(f.fn, 65536, 65537), std::invalid_argument);
        CHECK(!file_exists(f.fn));
        CHECK_THROWS_AS(quadriga_lib::hdf5_create_file(f.fn, 65536, 65536, 65536, 65536), std::invalid_argument);
        CHECK(!file_exists(f.fn));
    }
    SECTION("file cannot be created")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_create_file("test_hdf5_no_such_dir/x.hdf5", 2), std::runtime_error);
    }
}

TEST_CASE("hdf5_read_layout - channel IDs mark the occupied slots in write order")
{
    test_file f("test_hdf5_read_layout.hdf5");

    // A missing file gives a zero layout without an error
    arma::u32_vec layout = quadriga_lib::hdf5_read_layout(f.fn);
    REQUIRE(layout.n_elem == 4);
    CHECK(arma::all(layout == 0));

    quadriga_lib::hdf5_create_file(f.fn, 5);
    auto c = small_channel("c");
    quadriga_lib::hdf5_write_channel(f.fn, c, 3);
    quadriga_lib::hdf5_write_channel(f.fn, c, 0);
    std::any a = 1.0;
    quadriga_lib::hdf5_write_dset(f.fn, "a", a, 4);

    // New IDs count up from the largest ID in the file
    arma::u32_vec id(9, arma::fill::value(7)); // Stale content of another size is replaced
    layout = quadriga_lib::hdf5_read_layout(f.fn, &id);
    CHECK(layout(0) == 5);
    REQUIRE(id.n_elem == 5);
    CHECK(id(0) == 2);
    CHECK(id(1) == 0);
    CHECK(id(2) == 0);
    CHECK(id(3) == 1);
    CHECK(id(4) == 3);
}

TEST_CASE("hdf5_read_layout - input validation")
{
    test_file f("test_hdf5_read_layout_validation.txt");
    quadriga_lib::hdf5_create_file(f.fn, 2);
    CHECK_NOTHROW(quadriga_lib::hdf5_read_layout(f.fn));

    SECTION("not an HDF5 file")
    {
        std::ofstream(f.fn) << "This is not HDF5 data";
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_layout(f.fn), std::runtime_error);
    }
}

TEST_CASE("hdf5_reshape_layout - changes the layout and keeps the slots in linear order")
{
    test_file f("test_hdf5_reshape.hdf5");

    quadriga_lib::hdf5_create_file(f.fn, 6);
    auto a = small_channel("slot 1"), b = small_channel("slot 4");
    quadriga_lib::hdf5_write_channel(f.fn, a, 1);
    quadriga_lib::hdf5_write_channel(f.fn, b, 4);
    const arma::u32_vec id_before = channel_ids(f.fn);

    quadriga_lib::hdf5_reshape_layout(f.fn, 2, 3);
    arma::u32_vec id;
    arma::u32_vec layout = quadriga_lib::hdf5_read_layout(f.fn, &id);
    CHECK(layout(0) == 2);
    CHECK(layout(1) == 3);
    CHECK(layout(2) == 1);
    CHECK(layout(3) == 1);
    CHECK(arma::all(id == id_before));

    // Linear slot 1 is (1, 0), slot 4 is (0, 2)
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 1, 0).name == "slot 1");
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 0, 2).name == "slot 4");
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 0, 1).empty());

    // Default layout reshaped to 256 x 256; writing outside the new layout throws, inside works
    std::remove(f.fn.c_str());
    quadriga_lib::hdf5_create_file(f.fn);
    quadriga_lib::hdf5_reshape_layout(f.fn, 256, 256);
    layout = quadriga_lib::hdf5_read_layout(f.fn);
    CHECK(layout(0) == 256);
    CHECK(layout(1) == 256);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, a, 0, 256), std::invalid_argument);
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, a, 255, 255) == 0);
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 255, 255).coeff_re.size() == 1);
}

TEST_CASE("hdf5_reshape_layout - input validation")
{
    test_file f("test_hdf5_reshape_validation.hdf5");
    quadriga_lib::hdf5_create_file(f.fn); // 65536 slots
    CHECK_NOTHROW(quadriga_lib::hdf5_reshape_layout(f.fn, 65536));

    SECTION("missing file")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_reshape_layout("test_hdf5_reshape_missing.hdf5", 1), std::runtime_error);
    }
    SECTION("not an HDF5 file")
    {
        std::ofstream(f.fn) << "This is not HDF5 data";
        CHECK_THROWS_AS(quadriga_lib::hdf5_reshape_layout(f.fn, 65536), std::runtime_error);
    }
    SECTION("different number of slots, the layout is unchanged")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_reshape_layout(f.fn, 65535), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_reshape_layout(f.fn, 0, 65536), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_reshape_layout(f.fn, 65536, 65537), std::invalid_argument); // > 2^32 - 1
        arma::u32_vec id;
        arma::u32_vec layout = quadriga_lib::hdf5_read_layout(f.fn, &id);
        CHECK(layout(0) == 65536);
        CHECK(layout(1) == 1);
        CHECK(id.n_elem == 65536);
    }
}

// ---- hdf5_write_channel / hdf5_read_channel ----

TEST_CASE("hdf5_write_channel - slot (ix, iy, iz, iw) is the linear index ix + nx * (iy + ny * (iz + nz * iw))")
{
    test_file f("test_hdf5_write_index.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2, 3, 2, 2);

    const unsigned slots[3][4] = {{1, 2, 1, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}};
    for (auto &s : slots)
    {
        auto c = small_channel("slot " + std::to_string(s[0] + 2 * (s[1] + 3 * (s[2] + 2 * s[3]))));
        CHECK(quadriga_lib::hdf5_write_channel(f.fn, c, s[0], s[1], s[2], s[3]) == 0);
        std::any v = (unsigned)s[3];
        quadriga_lib::hdf5_write_dset(f.fn, "w", v, s[0], s[1], s[2], s[3]);
    }

    const arma::u32_vec id = channel_ids(f.fn);
    REQUIRE(id.n_elem == 24);
    CHECK(id(23) == 1);
    CHECK(id(14) == 2);
    CHECK(id(7) == 3);
    CHECK(arma::accu(id != 0) == 3);
    for (auto &s : slots)
    {
        const unsigned linear = s[0] + 2 * (s[1] + 3 * (s[2] + 2 * s[3]));
        INFO("linear slot " << linear);
        CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, s[0], s[1], s[2], s[3]).name == "slot " + std::to_string(linear));
        std::any v = quadriga_lib::hdf5_read_dset(f.fn, "w", s[0], s[1], s[2], s[3]);
        CHECK(std::any_cast<unsigned>(v) == s[3]);
        std::vector<std::string> names;
        names = quadriga_lib::hdf5_read_dset_names(f.fn, s[0], s[1], s[2], s[3]);
        CHECK(names.size() == 1);
    }
}

TEST_CASE("hdf5_write_channel - structured data round trips in single precision, float and double")
{
    test_file f("test_hdf5_write_round_trip.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2); // Two slots: a write that creates the file would make only the one it uses

    SECTION("one snapshot")
    {
        check_round_trip(f.fn, full_channel<double>({5}));
        check_round_trip(f.fn, full_channel<float>({5}), 1);
    }
    SECTION("three snapshots, positions and center frequency per snapshot")
    {
        check_round_trip(f.fn, full_channel<double>({4, 6, 5}));
        check_round_trip(f.fn, full_channel<float>({4, 6, 5}), 1);
    }
    SECTION("three snapshots, one position and center frequency for all")
    {
        check_round_trip(f.fn, full_channel<double>({4, 6, 5}, false));
        check_round_trip(f.fn, full_channel<float>({4, 6, 5}, false), 1);
    }
    SECTION("float and double reads agree within float accuracy")
    {
        auto c = full_channel<double>({3, 2});
        quadriga_lib::hdf5_write_channel(f.fn, c);
        auto rf = quadriga_lib::hdf5_read_channel<float>(f.fn);
        auto rd = quadriga_lib::hdf5_read_channel<double>(f.fn);
        REQUIRE(rf.coeff_re.size() == 2);
        REQUIRE(rd.coeff_re.size() == 2);
        for (size_t s = 0; s < 2; ++s)
        {
            CHECK(arma::approx_equal(arma::conv_to<arma::cube>::from(rf.coeff_re[s]), rd.coeff_re[s], "absdiff", 0.0));
            CHECK(arma::approx_equal(rd.coeff_re[s], c.coeff_re[s], "reldiff", 1e-7)); // float has 24 bits
            CHECK(arma::approx_equal(rd.delay[s], c.delay[s], "reldiff", 1e-7));
        }
    }
}

TEST_CASE("hdf5_write_channel - snapshots without paths keep their place and shape")
{
    test_file f("test_hdf5_write_empty_snapshots.hdf5");

    check_round_trip(f.fn, full_channel<double>({0}));       // 1 snapshot without paths
    check_round_trip(f.fn, full_channel<double>({3, 0}));    // trailing empty snapshot, delay [1, 1, 0]
    check_round_trip(f.fn, full_channel<double>({4, 0, 5})); // empty snapshot in the middle
    check_round_trip(f.fn, full_channel<float>({0, 0, 2}));  // leading empty snapshots
    check_round_trip(f.fn, full_channel<double>({0, 0}));    // no paths at all

    auto c = full_channel<double>({2, 3});
    for (auto &n_interact : c.n_interact) // Only LOS paths: interaction coordinates of size [3, 0]
        n_interact.zeros();
    for (auto &interact_coord : c.interact_coord)
        interact_coord.set_size(3, 0);
    check_round_trip(f.fn, c);
}

TEST_CASE("hdf5_write_channel - channels with a subset of the fields")
{
    test_file f("test_hdf5_write_subset.hdf5");

    SECTION("no center frequency and no orientations")
    {
        auto c = full_channel<double>({2, 3});
        c.center_freq.reset();
        c.tx_orientation.reset();
        c.rx_orientation.reset();
        check_round_trip(f.fn, c);
    }
    SECTION("path data without coefficients")
    {
        quadriga_lib::channel<double> c;
        c.tx_pos = filled(arma::mat(3, 1), 1.0);
        c.rx_pos = filled(arma::mat(3, 2), 2.0);
        c.delay = {filled(arma::cube(1, 1, 2), 3.0, 1e-9), filled(arma::cube(1, 1, 1), 4.0, 1e-9)};
        c.path_gain = {filled(arma::vec(2), 5.0), filled(arma::vec(1), 6.0)};
        check_round_trip(f.fn, c);
    }
    SECTION("positions only, two snapshots")
    {
        quadriga_lib::channel<double> c;
        c.tx_pos = filled(arma::mat(3, 1), 1.0);
        c.rx_pos = filled(arma::mat(3, 2), 2.0);
        c.initial_position = 1;
        check_round_trip(f.fn, c);
    }
    SECTION("only the initial position, no snapshots")
    {
        quadriga_lib::channel<double> c;
        c.initial_position = 5;
        CHECK(quadriga_lib::hdf5_write_channel(f.fn, c) == 0);
        auto r = quadriga_lib::hdf5_read_channel<double>(f.fn);
        CHECK(r.initial_position == 5);
        CHECK(r.n_snap() == 0);
        CHECK(r.name == c.name);
    }
}

TEST_CASE("hdf5_write_channel - a channel generated by the library round trips")
{
    test_file f("test_hdf5_write_generated.hdf5");

    // Test antenna: three omni elements with different positions and gains
    const double center_freq = 21.0e6;
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    quadriga_lib::arrayant_copy_element(ant, 0, {1});
    quadriga_lib::arrayant_copy_element(ant, 0, {2});
    ant.element_pos(1, 0) = -1.0;
    ant.element_pos(1, 1) = 0.0;
    ant.element_pos(1, 2) = 1.0;
    ant.e_theta_re.slice(1) *= 2;
    ant.e_theta_re.slice(1) *= 3;

    auto c = quadriga_lib::channel<double>();
    CHECK(c.empty());
    c.name = "Very cool test channel object with a long name!!!";
    c.center_freq = center_freq;
    c.tx_pos = arma::vec{0.0, 0.0, 1.0};
    c.rx_pos = {{20.0, 20.0}, {0.0, 1.0}, {1.0, 1.0}};
    c.tx_orientation = arma::mat(3, 1, arma::fill::zeros);
    c.rx_orientation = arma::mat(3, 1, arma::fill::zeros);

    // Two snapshots: a LOS path and a path with one interaction at (0, 10, 11)
    for (arma::uword s = 0; s < 2; ++s)
    {
        c.path_gain.push_back({1.0, 0.25});
        c.polmat.push_back(arma::mat(8, 2, arma::fill::zeros));
        c.polmat[s].row(0).fill(1.0);
        c.polmat[s].row(6).fill(-1.0);
        c.n_interact.push_back({0, 1});
        c.interact_coord.push_back(arma::mat({0.0, 10.0, 11.0}).t());
        c.path_length.push_back(arma::vec());
        c.path_angles.push_back(arma::mat());
        c.coeff_re.push_back(arma::cube());
        c.coeff_im.push_back(arma::cube());
        c.delay.push_back(arma::cube());

        arma::mat fbs_pos, lbs_pos;
        std::vector<arma::mat> path_coord;
        quadriga_lib::coord2path({c.tx_pos(0), c.tx_pos(1), c.tx_pos(2)}, {c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s)},
                                 c.n_interact[s], c.interact_coord[s], &c.path_length[s], &fbs_pos, &lbs_pos,
                                 &c.path_angles[s], &path_coord);
        if (s == 0) // Fixture check: the LOS path runs from TX to RX
        {
            arma::mat T = {{0.0, 20.0}, {0.0, 0.0}, {1.0, 1.0}};
            CHECK(arma::approx_equal(T, path_coord[0], "absdiff", 1e-13));
        }
        quadriga_lib::get_channels_spherical<double>(ant, ant, {c.tx_pos(0), c.tx_pos(1), c.tx_pos(2)}, {0.0, 0.0, 0.0},
                                                     {c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s)}, {0.0, 0.0, 0.0},
                                                     fbs_pos, fbs_pos, c.path_gain[s], c.path_length[s], c.polmat[s],
                                                     c.coeff_re[s], c.coeff_im[s], c.delay[s], center_freq, true, false);
    }

    quadriga_lib::hdf5_create_file(f.fn, 10);
    quadriga_lib::hdf5_write_channel(f.fn, c, 7);
    auto d = quadriga_lib::hdf5_read_channel<double>(f.fn, 7);
    CHECK(!d.empty());
    check_structured(d, c);
}

TEST_CASE("hdf5_write_channel - channel names are stored with at most 255 characters")
{
    test_file f("test_hdf5_write_name.hdf5");

    for (size_t len : {0, 1, 20, 255})
    {
        INFO("length " << len);
        auto c = small_channel(std::string(len, 'N'));
        quadriga_lib::hdf5_write_channel(f.fn, c);
        CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn).name == c.name);
    }

    auto c = small_channel("\xc3\xa4\xc3\xb6 UTF-8 bytes"); // Bytes are kept as they are
    quadriga_lib::hdf5_write_channel(f.fn, c);
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn).name == c.name);

    // A name of 300 characters is rejected by the validation
    std::remove(f.fn.c_str());
    c.name = std::string(300, 'L');
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, c), std::invalid_argument);
    CHECK(!file_exists(f.fn));
}

TEST_CASE("hdf5_write_channel - unstructured data of every supported type round trips")
{
    test_file f("test_hdf5_write_par_types.hdf5");

    // Slot 1: structured data only; slot 7: a channel with all types; slot 0: the same datasets one by one
    quadriga_lib::hdf5_create_file(f.fn, 10);
    auto c = small_channel("structured");
    quadriga_lib::hdf5_write_channel(f.fn, c, 1);
    c.name = "with unstructured data";
    all_par_types(c.par_names, c.par_data);
    REQUIRE(c.par_names.size() == 31);
    quadriga_lib::hdf5_write_channel(f.fn, c, 7);
    for (size_t i = 0; i < c.par_names.size(); ++i)
        quadriga_lib::hdf5_write_dset(f.fn, c.par_names[i], c.par_data[i]);

    const arma::u32_vec id = channel_ids(f.fn);
    CHECK(id(0) == 3);
    CHECK(id(1) == 1);
    CHECK(id(7) == 2);
    CHECK(id(5) == 0);

    for (unsigned slot : {0, 7})
    {
        INFO("slot " << slot);
        std::vector<std::string> names;
        names = quadriga_lib::hdf5_read_dset_names(f.fn, slot);
        CHECK(names.size() == c.par_names.size());
        CHECK(names == c.par_names); // In write order
        for (size_t i = 0; i < c.par_names.size(); ++i)
        {
            INFO("dataset " << c.par_names[i]);
            CHECK(same_any(c.par_data[i], quadriga_lib::hdf5_read_dset(f.fn, c.par_names[i], slot)));
        }
    }

    // hdf5_read_channel returns the unstructured data in write order, together with the structured data
    auto d = quadriga_lib::hdf5_read_channel<double>(f.fn, 7);
    CHECK(d.is_valid().empty());
    CHECK(d.name == c.name);
    CHECK(d.coeff_re.size() == 1);
    CHECK(d.par_names == c.par_names);
    REQUIRE(d.par_data.size() == c.par_data.size());
    for (size_t i = 0; i < c.par_names.size(); ++i)
    {
        INFO("dataset " << c.par_names[i]);
        CHECK(same_any(c.par_data[i], d.par_data[i]));
    }

    // Unstructured data is not converted to the read type
    auto e = quadriga_lib::hdf5_read_channel<float>(f.fn, 0);
    REQUIRE(e.par_data.size() == c.par_data.size());
    CHECK(quadriga_lib::any_type_id(&e.par_data[2]) == 11); // scalar_double
    CHECK(quadriga_lib::any_type_id(&e.par_data[8]) == 21); // mat_double
}

TEST_CASE("hdf5_write_dset - scalars, single elements, empty arrays and strings keep their shape")
{
    test_file f("test_hdf5_write_dset_shapes.hdf5");

    // A scalar gives a scalar, a vector an arma::Col (also with a single element), a matrix an arma::Mat, a cube an
    // arma::Cube; strings come back without trailing NUL bytes
    const std::vector<std::pair<std::string, std::any>> par = {
        {"s_double", 2.5}, {"vec1", arma::vec{7.0}}, {"fvec1", arma::fvec{7.0f}}, {"u32vec1", arma::Col<unsigned>{5u}},
        {"ivec1", arma::ivec{-9}}, {"row1", arma::rowvec{7.0}}, {"mat11", arma::mat(1, 1, arma::fill::ones)},
        {"cube111", arma::cube(1, 1, 1, arma::fill::ones)}, {"str_empty", std::string("")}, {"str", std::string("abc")},
        {"str_inner_nul", std::string("ab\0cd", 5)}, {"mat0x3", arma::mat(0, 3)}, {"mat3x0", arma::Mat<int>(3, 0)},
        {"cube230", arma::cube(2, 3, 0)}, {"vec0", arma::vec()}, {"row0", arma::urowvec()}, {std::string(260, 'L'), 1.0}};
    for (auto &p : par)
        quadriga_lib::hdf5_write_dset(f.fn, p.first, p.second);

    std::vector<std::string> names;
    names = quadriga_lib::hdf5_read_dset_names(f.fn);
    CHECK(names.size() == par.size());
    REQUIRE(names.size() == par.size());
    for (size_t i = 0; i < par.size(); ++i)
    {
        INFO("dataset " << par[i].first);
        CHECK(names[i] == par[i].first);
        CHECK(same_any(par[i].second, quadriga_lib::hdf5_read_dset(f.fn, par[i].first)));
    }
    CHECK(std::any_cast<std::string>(quadriga_lib::hdf5_read_dset(f.fn, "str_empty")).empty());
    CHECK(std::any_cast<arma::vec>(quadriga_lib::hdf5_read_dset(f.fn, "vec1"))(0) == 7.0);

    // Trailing NUL bytes of a string are not kept
    std::any trailing = std::string("ab\0\0", 4);
    quadriga_lib::hdf5_write_dset(f.fn, "str_trailing_nul", trailing);
    CHECK(std::any_cast<std::string>(quadriga_lib::hdf5_read_dset(f.fn, "str_trailing_nul")) == "ab");

    auto r = quadriga_lib::hdf5_read_channel<double>(f.fn);
    CHECK(r.par_names.size() == par.size() + 1);
}

TEST_CASE("hdf5_write_channel - overwriting a slot replaces it completely")
{
    test_file f("test_hdf5_write_overwrite.hdf5");

    // A dataset written with hdf5_write_dset, then a channel with dataset "b" into the same slot
    std::any a = 1.0;
    quadriga_lib::hdf5_write_dset(f.fn, "a", a);
    auto c = small_channel("c", 3);
    c.par_names = {"b"};
    c.par_data = {std::any(2.0)};
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, c) == 1);
    {
        auto r = quadriga_lib::hdf5_read_channel<double>(f.fn);
        REQUIRE(r.par_names.size() == 1);
        CHECK(r.par_names[0] == "b");
        CHECK(std::any_cast<double>(r.par_data[0]) == 2.0);
        CHECK(r.coeff_re.size() == 1);
        CHECK(!quadriga_lib::hdf5_read_dset(f.fn, "a").has_value());
        std::vector<std::string> names;
        names = quadriga_lib::hdf5_read_dset_names(f.fn);
        CHECK(names.size() == 1);
        REQUIRE(names.size() == 1);
        CHECK(names[0] == "b");
    }

    // A channel with unstructured data replaced by one without: no unstructured data is left
    auto plain = small_channel("plain");
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, plain) == 1);
    CHECK(quadriga_lib::hdf5_read_dset_names(f.fn).empty());
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn).par_names.empty());

    // A channel with structured data replaced by a parameter-only channel, written twice
    quadriga_lib::channel<double> p;
    p.par_names = {"x"};
    p.par_data = {std::any(1.0)};
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, p) == 1);
    p.par_data = {std::any(99.0)};
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, p) == 1);
    {
        auto r = quadriga_lib::hdf5_read_channel<double>(f.fn);
        CHECK(r.n_snap() == 0);
        CHECK(r.coeff_re.empty());
        REQUIRE(r.par_names.size() == 1);
        CHECK(std::any_cast<double>(r.par_data[0]) == 99.0);
    }

    // Five rewrites of one slot; the other slot is kept; every rewrite gets a new channel ID
    std::remove(f.fn.c_str());
    quadriga_lib::hdf5_create_file(f.fn, 4);
    auto other = small_channel("other");
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, other, 1) == 0);
    for (int i = 0; i < 5; ++i)
    {
        c.name = "rewrite " + std::to_string(i);
        CHECK(quadriga_lib::hdf5_write_channel(f.fn, c, 0) == (i == 0 ? 0 : 1));
    }
    arma::u32_vec id = channel_ids(f.fn);
    CHECK(id(0) == 6);
    CHECK(id(1) == 1);
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 0).name == "rewrite 4");
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 1).name == "other");
}

TEST_CASE("hdf5_write_channel - a channel without structured data")
{
    test_file f("test_hdf5_write_no_structured.hdf5");

    // Nothing to write: returns 0 and does not create the file
    quadriga_lib::channel<double> e;
    REQUIRE(e.empty());
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, e) == 0);
    CHECK(!file_exists(f.fn));

    // Unstructured data only: the slot holds the datasets; the channel name is not stored
    quadriga_lib::channel<double> p;
    p.name = "parameters";
    p.par_names = {"x"};
    p.par_data = {std::any(1.0)};
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, p, 2) == 0);
    CHECK(arma::all(quadriga_lib::hdf5_read_layout(f.fn) == arma::u32_vec({3, 1, 1, 1}))); // Just holds the slot
    auto r = quadriga_lib::hdf5_read_channel<double>(f.fn, 2);
    CHECK(r.empty());
    CHECK(r.name.empty());
    CHECK(r.initial_position == 0);
    REQUIRE(r.par_names.size() == 1);
    CHECK(r.par_names[0] == "x");
    CHECK(channel_ids(f.fn)(2) == 1);

    // The empty channel leaves the existing file and its slots unchanged
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, e, 1) == 0);
    CHECK(channel_ids(f.fn)(1) == 0);
}

TEST_CASE("hdf5_write_channel - an invalid channel leaves the file unchanged")
{
    test_file f("test_hdf5_write_invalid_channel.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 4);
    auto good = small_channel("good");
    quadriga_lib::hdf5_write_channel(f.fn, good, 0);

    // Invalid channels are rejected before the file is changed: no slot is written, no channel ID is used up
    auto coeff = small_channel("coeff");
    coeff.coeff_im[0].set_size(2, 3, 3);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, coeff, 1), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, coeff, 0), std::invalid_argument);

    auto pos = small_channel("pos");
    pos.tx_pos.set_size(2, 1);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, pos, 1), std::invalid_argument);

    auto par = small_channel("par");
    par.par_names = {"a"}; // Without data
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, par, 1), std::invalid_argument);
    par.par_data = {std::any(arma::cx_mat(2, 2))}; // Unsupported type
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, par, 1), std::invalid_argument);

    arma::u32_vec id = channel_ids(f.fn);
    CHECK(id(0) == 1);
    CHECK(id(1) == 0);
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 0).name == "good");
    CHECK(quadriga_lib::hdf5_write_channel(f.fn, good, 1) == 0);
    CHECK(channel_ids(f.fn)(1) == 2);
}

TEST_CASE("hdf5_write_channel - input validation")
{
    test_file f("test_hdf5_write_validation.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2, 2, 2, 2);
    auto c = small_channel("valid");
    c.par_names = {"a"};
    c.par_data = {std::any(1.0)};
    CHECK_NOTHROW(quadriga_lib::hdf5_write_channel(f.fn, c, 1, 1, 1, 1));
    const arma::u32_vec id_before = channel_ids(f.fn);

    SECTION("invalid channel, the file is not created")
    {
        test_file g("test_hdf5_write_validation_new.hdf5");
        auto bad = c;
        bad.delay[0].set_size(1, 3, 2);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(g.fn, bad), std::invalid_argument);
        CHECK(!file_exists(g.fn));
    }
    SECTION("par_names and par_data differ in length")
    {
        auto bad = c;
        bad.par_names.push_back("b");
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, bad), std::invalid_argument);
    }
    SECTION("invalid dataset name")
    {
        for (const char *name : {"bad name", "bad-name", "\xc3\xa4", ""})
        {
            INFO("name '" << name << "'");
            auto bad = c;
            bad.par_names = {name};
            CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, bad), std::invalid_argument);
        }
    }
    SECTION("duplicate dataset names, the slot is unchanged")
    {
        auto bad = small_channel("duplicate");
        bad.par_names = {"a", "a"};
        bad.par_data = {std::any(1.0), std::any(2.0)};
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, bad, 1, 1, 1, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, bad, 0), std::invalid_argument);
        CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 1, 1, 1, 1).name == "valid");
    }
    SECTION("index out of range in each dimension")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, c, 2, 0, 0, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, c, 0, 2, 0, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, c, 0, 0, 2, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, c, 0, 0, 0, 2), std::invalid_argument);
    }
    SECTION("not an HDF5 file")
    {
        test_file g("test_hdf5_write_validation.txt");
        std::ofstream(g.fn) << "This is not HDF5 data";
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(g.fn, c), std::runtime_error);
    }
    CHECK(arma::all(channel_ids(f.fn) == id_before));
}

TEST_CASE("hdf5_read_channel - empty slots and indices outside the layout give an empty channel")
{
    test_file f("test_hdf5_read_channel_empty.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2, 2);
    auto c = small_channel("c");
    quadriga_lib::hdf5_write_channel(f.fn, c, 1, 1);

    for (auto i : {std::vector<unsigned>{0, 0, 0, 0}, {2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}})
    {
        INFO("index " << i[0] << " " << i[1] << " " << i[2] << " " << i[3]);
        CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, i[0], i[1], i[2], i[3]).empty());
        CHECK(quadriga_lib::hdf5_read_channel<float>(f.fn, i[0], i[1], i[2], i[3]).empty());
    }
    CHECK(!quadriga_lib::hdf5_read_channel<float>(f.fn, 1, 1).empty());
}

TEST_CASE("hdf5_read_channel - input validation")
{
    test_file f("test_hdf5_read_channel_validation.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2);
    CHECK_NOTHROW(quadriga_lib::hdf5_read_channel<double>(f.fn));

    SECTION("missing file")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>("test_hdf5_read_channel_missing.hdf5"), std::runtime_error);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<float>("test_hdf5_read_channel_missing.hdf5"), std::runtime_error);
    }
    SECTION("not an HDF5 file")
    {
        std::ofstream(f.fn) << "This is not HDF5 data";
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(f.fn), std::runtime_error);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<float>(f.fn), std::runtime_error);
    }
}

// ---- hdf5_write_dset / hdf5_read_dset / hdf5_read_dset_names ----

TEST_CASE("hdf5_write_dset - adds a dataset to an empty or a filled slot")
{
    test_file f("test_hdf5_write_dset.hdf5");

    // A missing file is created just large enough for the slot
    std::any a = arma::vec{1.0, 2.0};
    quadriga_lib::hdf5_write_dset(f.fn, "a", a, 3);
    CHECK(arma::all(quadriga_lib::hdf5_read_layout(f.fn) == arma::u32_vec({4, 1, 1, 1})));
    CHECK(channel_ids(f.fn)(3) == 1);

    // Empty slot: a new channel ID; the slot holds only the dataset
    auto r = quadriga_lib::hdf5_read_channel<double>(f.fn, 3);
    CHECK(r.empty());
    REQUIRE(r.par_names.size() == 1);
    CHECK(r.par_names[0] == "a");

    // Filled slot without unstructured data: structured data is kept, the channel ID does not change
    auto c = full_channel<double>({2, 3});
    quadriga_lib::hdf5_write_channel(f.fn, c, 1);
    quadriga_lib::hdf5_write_dset(f.fn, "x", a, 1);
    CHECK(channel_ids(f.fn)(1) == 2);
    r = quadriga_lib::hdf5_read_channel<double>(f.fn, 1);
    REQUIRE(r.par_names.size() == 1);
    CHECK(r.par_names[0] == "x");
    r.par_names.clear(), r.par_data.clear();
    check_structured(r, c);

    // Filled slot with unstructured data: appended after the existing datasets
    c.par_names = {"p", "q"};
    c.par_data = {std::any(1), std::any(2)};
    quadriga_lib::hdf5_write_channel(f.fn, c, 2);
    quadriga_lib::hdf5_write_dset(f.fn, "extra", a, 2);
    CHECK(channel_ids(f.fn)(2) == 3);
    r = quadriga_lib::hdf5_read_channel<double>(f.fn, 2);
    CHECK(r.par_names == std::vector<std::string>({"p", "q", "extra"}));
    CHECK(r.coeff_re.size() == 2);

    // A custom prefix: listed and read with that prefix only; hdf5_read_channel reads only "par_" datasets
    std::any b = 7;
    quadriga_lib::hdf5_write_dset(f.fn, "y", b, 2, 0, 0, 0, "xyz_");
    std::vector<std::string> names;
    names = quadriga_lib::hdf5_read_dset_names(f.fn, 2, 0, 0, 0, "xyz_");
    CHECK(names.size() == 1);
    CHECK(names == std::vector<std::string>({"y"}));
    CHECK(std::any_cast<int>(quadriga_lib::hdf5_read_dset(f.fn, "y", 2, 0, 0, 0, "xyz_")) == 7);
    CHECK(!quadriga_lib::hdf5_read_dset(f.fn, "y", 2).has_value());
    CHECK(quadriga_lib::hdf5_read_dset_names(f.fn, 2).size() == 3);
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 2).par_names.size() == 3);
}

TEST_CASE("hdf5_write_dset - input validation")
{
    test_file f("test_hdf5_write_dset_validation.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 4);
    std::any v = arma::vec{1.0, 2.0};
    CHECK_NOTHROW(quadriga_lib::hdf5_write_dset(f.fn, "valid", v, 1));
    const arma::u32_vec id_before = channel_ids(f.fn);

    SECTION("invalid name or prefix, the empty slot is not reserved")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "bad name", v, 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "bad-name", v, 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "a", v, 2, 0, 0, 0, "bad prefix"), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "", v, 2), std::invalid_argument); // with the default prefix
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "", v, 2, 0, 0, 0, ""), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "par_", v, 2, 0, 0, 0, ""), std::invalid_argument); // empty par name
    }
    SECTION("no data")
    {
        std::any none;
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "a", none, 2), std::invalid_argument);
    }
    SECTION("unsupported types")
    {
        for (std::any u : {std::any(arma::cx_mat(2, 2)), std::any(std::vector<int>{1}), std::any('c'),
                           std::any((const char *)"abc"), std::any((unsigned short)3)})
        {
            INFO("type " << u.type().name());
            CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "u", u, 2), std::invalid_argument);
        }
    }
    SECTION("existing name, no overwrite")
    {
        std::any w = 1.0;
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "valid", w, 1), std::runtime_error);
        CHECK(std::any_cast<arma::vec>(quadriga_lib::hdf5_read_dset(f.fn, "valid", 1)).n_elem == 2);
    }
    SECTION("index out of range in each dimension")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "a", v, 4, 0, 0, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "a", v, 0, 1, 0, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "a", v, 0, 0, 1, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "a", v, 0, 0, 0, 1), std::invalid_argument);
    }
    SECTION("not an HDF5 file")
    {
        test_file g("test_hdf5_write_dset_validation.txt");
        std::ofstream(g.fn) << "This is not HDF5 data";
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(g.fn, "a", v), std::runtime_error);
    }
    CHECK(arma::all(channel_ids(f.fn) == id_before));
    CHECK(quadriga_lib::hdf5_read_dset_names(f.fn, 1).size() == 1);
}

TEST_CASE("hdf5_read_dset - missing slots, groups and names give an empty std::any")
{
    test_file f("test_hdf5_read_dset_missing.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 3);
    auto c = small_channel("no unstructured data");
    quadriga_lib::hdf5_write_channel(f.fn, c, 1);
    std::any v = 1.0;
    quadriga_lib::hdf5_write_dset(f.fn, "a", v, 2);

    CHECK(!quadriga_lib::hdf5_read_dset(f.fn, "a", 0).has_value()); // empty slot
    CHECK(!quadriga_lib::hdf5_read_dset(f.fn, "a", 1).has_value()); // slot without unstructured data
    CHECK(!quadriga_lib::hdf5_read_dset(f.fn, "b", 2).has_value()); // other name
    CHECK(!quadriga_lib::hdf5_read_dset(f.fn, "a", 3).has_value()); // outside the layout
    CHECK(!quadriga_lib::hdf5_read_dset(f.fn, "a", 2, 0, 0, 0, "x_").has_value());
    CHECK(quadriga_lib::hdf5_read_dset(f.fn, "a", 2).has_value());
}

TEST_CASE("hdf5_read_dset - input validation")
{
    test_file f("test_hdf5_read_dset_validation.hdf5");
    std::any v = 1.0;
    quadriga_lib::hdf5_write_dset(f.fn, "a", v);
    CHECK_NOTHROW(quadriga_lib::hdf5_read_dset(f.fn, "a"));

    SECTION("missing file")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset("test_hdf5_read_dset_missing.hdf5", "a"), std::runtime_error);
    }
    SECTION("not an HDF5 file")
    {
        std::ofstream(f.fn) << "This is not HDF5 data";
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset(f.fn, "a"), std::runtime_error);
    }
}

TEST_CASE("hdf5_read_dset_names - lists the names after the prefix in write order")
{
    test_file f("test_hdf5_read_dset_names.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 4);
    for (const char *name : {"zeta", "alpha", "s_one", "mid", "s_two"})
    {
        std::any v = 1;
        quadriga_lib::hdf5_write_dset(f.fn, name, v, 1);
    }
    auto c = small_channel("no unstructured data");
    quadriga_lib::hdf5_write_channel(f.fn, c, 2);

    auto names = quadriga_lib::hdf5_read_dset_names(f.fn, 1);
    CHECK(names == std::vector<std::string>({"zeta", "alpha", "s_one", "mid", "s_two"}));

    // A longer prefix selects a subset; an empty prefix lists the full dataset names
    names = quadriga_lib::hdf5_read_dset_names(f.fn, 1, 0, 0, 0, "par_s_");
    CHECK(names == std::vector<std::string>({"one", "two"}));
    names = quadriga_lib::hdf5_read_dset_names(f.fn, 1, 0, 0, 0, "");
    CHECK(names.size() == 5);
    CHECK(names.front() == "par_zeta");

    // Empty slot (default slot 0), slot without unstructured data, index outside the layout
    CHECK(quadriga_lib::hdf5_read_dset_names(f.fn).empty());
    CHECK(quadriga_lib::hdf5_read_dset_names(f.fn, 2).empty());
    CHECK(quadriga_lib::hdf5_read_dset_names(f.fn, 4).empty());
}

TEST_CASE("hdf5_read_dset_names - input validation")
{
    test_file f("test_hdf5_read_dset_names_validation.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2);
    CHECK_NOTHROW(quadriga_lib::hdf5_read_dset_names(f.fn));

    SECTION("missing file")
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset_names("test_hdf5_read_dset_names_missing.hdf5"), std::runtime_error);
    }
    SECTION("not an HDF5 file")
    {
        std::ofstream(f.fn) << "This is not HDF5 data";
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset_names(f.fn), std::runtime_error);
    }
}

TEST_CASE("hdf5_write_channel - files that cannot be opened for writing throw std::runtime_error")
{
    test_file f("test_hdf5_read_only.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2);
    namespace fs = std::filesystem;
    const auto write_bits = fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write;
    fs::permissions(f.fn, write_bits, fs::perm_options::remove);

    if (FILE *probe = std::fopen(f.fn.c_str(), "r+")) // E.g. running as root: read-only files stay writable
        std::fclose(probe);
    else
    {
        auto c = small_channel("c");
        std::any v = 1.0;
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel(f.fn, c), std::runtime_error);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(f.fn, "a", v), std::runtime_error);
        CHECK_THROWS_AS(quadriga_lib::hdf5_reshape_layout(f.fn, 1, 2), std::runtime_error);
        CHECK_THROWS_AS(quadriga_lib::hdf5_create_file(f.fn, 2), std::runtime_error);
        CHECK(arma::all(quadriga_lib::hdf5_read_layout(f.fn) == arma::u32_vec({2, 1, 1, 1})));
    }
    fs::permissions(f.fn, fs::perms::owner_write, fs::perm_options::add);
}

// ---- Snapshot selection (i_snap) and batch read / write of several slots ----

namespace
{
    // The channel 'c' reduced to the snapshots 'i_snap', as hdf5_read_channel returns it; values per snapshot are
    // sliced, values stored once per channel are kept
    quadriga_lib::channel<double> select_snapshots(const quadriga_lib::channel<double> &c, const arma::uvec &i_snap)
    {
        quadriga_lib::channel<double> r = c;
        const arma::uword n_snap = c.n_snap();
        auto pick = [&](auto &field)
        {
            if (field.empty())
                return;
            std::remove_reference_t<decltype(field)> out;
            for (arma::uword i : i_snap)
                out.push_back(field[i]);
            field = out;
        };
        pick(r.coeff_re), pick(r.coeff_im), pick(r.delay), pick(r.path_gain), pick(r.path_length);
        pick(r.polmat), pick(r.path_angles), pick(r.fbs_pos), pick(r.lbs_pos);
        pick(r.n_interact), pick(r.interact_coord);
        if (n_snap > 1 && c.center_freq.n_elem == n_snap)
            r.center_freq = c.center_freq.elem(i_snap);
        for (auto p : {&r.tx_pos, &r.rx_pos, &r.tx_orientation, &r.rx_orientation})
            if (n_snap > 1 && p->n_cols == n_snap)
                *p = arma::mat(p->cols(i_snap));
        return r;
    }

    std::string file_bytes(const std::string &fn)
    {
        std::ifstream f(fn, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
}

TEST_CASE("hdf5_read_channel - i_snap reads the selected snapshots")
{
    test_file f("test_hdf5_i_snap.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 5);

    auto per_snap = full_channel<double>({3, 0, 2, 4}); // Positions, orientations and frequency per snapshot
    per_snap.par_names = {"x"}, per_snap.par_data = {1.5};
    auto shared = full_channel<double>({2, 1, 3}, false); // One position for all snapshots
    auto single = full_channel<double>({5});
    quadriga_lib::channel<double> par_only;
    par_only.par_names = {"y"}, par_only.par_data = {2.5};
    quadriga_lib::hdf5_write_channel(f.fn, per_snap, 0);
    quadriga_lib::hdf5_write_channel(f.fn, shared, 1);
    quadriga_lib::hdf5_write_channel(f.fn, single, 2);
    quadriga_lib::hdf5_write_channel(f.fn, par_only, 3);

    for (arma::uvec i_snap : {arma::uvec{2}, arma::uvec{3, 0}, arma::uvec{1, 1, 2}, arma::uvec{0, 1, 2, 3}})
    {
        INFO("i_snap " << arma::conv_to<std::vector<arma::uword>>::from(i_snap)[0] << ", n = " << i_snap.n_elem);
        auto r = quadriga_lib::hdf5_read_channel<double>(f.fn, 0, 0, 0, 0, i_snap);
        check_structured(r, select_snapshots(per_snap, i_snap));
        CHECK(r.par_names == per_snap.par_names);
        check_structured(quadriga_lib::hdf5_read_channel<float>(f.fn, 0, 0, 0, 0, i_snap), select_snapshots(per_snap, i_snap));
    }
    arma::uvec i_shared = {2, 0}, i_single = {0, 0}, none;
    check_structured(quadriga_lib::hdf5_read_channel<double>(f.fn, 1, 0, 0, 0, i_shared), select_snapshots(shared, i_shared));
    check_structured(quadriga_lib::hdf5_read_channel<double>(f.fn, 2, 0, 0, 0, i_single), select_snapshots(single, i_single));
    check_structured(quadriga_lib::hdf5_read_channel<double>(f.fn, 0, 0, 0, 0, none), per_snap); // Empty = all

    // Index >= n_snap of a slot with snapshots throws; a slot without snapshots keeps its unstructured data; empty
    // slots and slots outside the layout stay empty
    arma::uvec i_4 = {4}, i_0 = {0}, i_7 = {7};
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(f.fn, 0, 0, 0, 0, i_4), std::invalid_argument);
    auto p = quadriga_lib::hdf5_read_channel<double>(f.fn, 3, 0, 0, 0, i_7);
    CHECK(p.n_snap() == 0);
    CHECK(p.par_names == par_only.par_names);
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 4, 0, 0, 0, i_7).empty());
    CHECK(quadriga_lib::hdf5_read_channel<double>(f.fn, 9, 0, 0, 0, i_7).empty());
}

TEST_CASE("hdf5_write_channel_vec / hdf5_read_channel_vec - equal loops of single calls")
{
    test_file f("test_hdf5_vec.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2, 2);
    std::vector<quadriga_lib::channel<double>> chan = {full_channel<double>({3, 1}), small_channel("b"), full_channel<double>({2})};
    const arma::u32_mat slots = {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 0}};

    CHECK(arma::all(quadriga_lib::hdf5_write_channel_vec(f.fn, chan, slots) == arma::s32_vec({0, 0, 0})));
    std::vector<quadriga_lib::channel<double>> again = {small_channel("c"), small_channel("d")};
    const arma::u32_mat new_and_old = {{0, 0}, {1, 0}, {0, 0}, {0, 0}}; // Slot (0, 1) is filled, (0, 0) too
    CHECK(arma::all(quadriga_lib::hdf5_write_channel_vec(f.fn, again, new_and_old) == arma::s32_vec({1, 1})));
    std::vector<quadriga_lib::channel<double>> last = {small_channel("e"), chan[2]};
    const arma::u32_mat empty_and_filled = {{1, 0}, {1, 0}, {0, 0}, {0, 0}};
    CHECK(arma::all(quadriga_lib::hdf5_write_channel_vec(f.fn, last, empty_and_filled) == arma::s32_vec({0, 1})));
    chan[1] = again[0];
    std::remove(f.fn.c_str()); // Back to the three channels written first, slot (1, 1) empty
    quadriga_lib::hdf5_create_file(f.fn, 2, 2);
    quadriga_lib::hdf5_write_channel_vec(f.fn, chan, slots);

    // All four slots of the layout and one outside it
    const arma::u32_mat all = {{0, 1, 0, 1, 2}, {0, 0, 1, 1, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    auto r = quadriga_lib::hdf5_read_channel_vec<double>(f.fn, all);
    REQUIRE(r.size() == 5);
    check_structured(r[0], chan[2]);
    check_structured(r[1], chan[0]);
    check_structured(r[2], chan[1]);
    CHECK(r[3].empty());
    CHECK(r[4].empty());
    for (arma::uword i = 0; i < 5; ++i)
        check_structured(r[i], quadriga_lib::hdf5_read_channel<double>(f.fn, all(0, i), all(1, i)));

    arma::uvec i_snap = {0};
    auto rs = quadriga_lib::hdf5_read_channel_vec<float>(f.fn, all, i_snap);
    check_structured(rs[1], select_snapshots(chan[0], i_snap));
    check_structured(rs[2], chan[1]);
    CHECK(rs[3].empty());
    CHECK(quadriga_lib::hdf5_read_channel_vec<double>(f.fn, arma::u32_mat()).empty());

    // The same slot twice in a file that holds only this channel (each read is limited by the file size)
    std::remove(f.fn.c_str());
    quadriga_lib::hdf5_create_file(f.fn, 1);
    quadriga_lib::hdf5_write_channel(f.fn, chan[0]);
    auto twice = quadriga_lib::hdf5_read_channel_vec<double>(f.fn, arma::u32_mat(4, 2, arma::fill::zeros));
    REQUIRE(twice.size() == 2);
    check_structured(twice[0], chan[0]);
    check_structured(twice[1], chan[0]);
}

TEST_CASE("hdf5_write_channel_vec / hdf5_read_channel_vec - input validation")
{
    test_file f("test_hdf5_vec_validation.hdf5");
    quadriga_lib::hdf5_create_file(f.fn, 2, 2);
    std::vector<quadriga_lib::channel<double>> two = {small_channel("a"), small_channel("b")};
    auto invalid = small_channel("x");
    invalid.coeff_im.clear();
    const arma::u32_mat col0 = {0, 0, 0, 0}, pair = {{0, 1}, {0, 0}, {0, 0}, {0, 0}};
    CHECK_NOTHROW(quadriga_lib::hdf5_read_channel_vec<double>(f.fn, pair));
    const std::string before = file_bytes(f.fn);

    // Nothing is written for any of these
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat(0, 2)), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat(col0.t())), std::invalid_argument); // 2 channels, 1 slot
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat(3, 2, arma::fill::zeros)), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat(4, 2, arma::fill::zeros)), std::invalid_argument); // Twice
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat({{0, 0}, {0, 2}, {0, 0}, {0, 0}})), std::invalid_argument);
    std::vector<quadriga_lib::channel<double>> bad = {small_channel("a"), invalid};
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, bad, pair), std::invalid_argument);
    CHECK(file_bytes(f.fn) == before);
    CHECK(quadriga_lib::hdf5_write_channel_vec(f.fn, std::vector<quadriga_lib::channel<double>>(), arma::u32_mat()).empty());

    // Missing file: created with the layout that just holds all slots; a layout that cannot be stored throws before
    // the file is created
    std::remove(f.fn.c_str());
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat({{0, UINT_MAX}, {0, 0}, {0, 0}, {0, 0}})), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat({{65535, 0}, {0, 65536}, {0, 0}, {0, 0}})), std::invalid_argument);
    CHECK(!file_exists(f.fn));
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel_vec<double>(f.fn, pair), std::runtime_error);
    CHECK(arma::all(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat({{0, 1}, {0, 2}, {0, 0}, {0, 1}})) == arma::s32_vec({0, 0})));
    CHECK(arma::all(quadriga_lib::hdf5_read_layout(f.fn) == arma::u32_vec({2, 3, 1, 2})));
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_channel_vec(f.fn, two, arma::u32_mat({{0, 2}, {0, 0}, {0, 0}, {0, 0}})), std::invalid_argument); // Outside an existing file
    std::remove(f.fn.c_str());
    CHECK(arma::all(quadriga_lib::hdf5_write_channel_vec(f.fn, two, pair) == arma::s32_vec({0, 0})));
    CHECK(arma::all(quadriga_lib::hdf5_read_layout(f.fn) == arma::u32_vec({2, 1, 1, 1})));
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel_vec<double>(f.fn, arma::u32_mat(3, 1, arma::fill::zeros)), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel_vec<double>(f.fn, arma::u32_mat(0, 2)), std::invalid_argument);
    const arma::uvec i_1 = {1}; // The written channels have one snapshot
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel_vec<double>(f.fn, pair, i_1), std::invalid_argument);
}
