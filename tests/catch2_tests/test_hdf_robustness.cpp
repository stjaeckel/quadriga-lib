// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Slot rewrites, malformed files, layout limits, name storage, round trips and side effects of the HDF5 functions.
// Files are patched with the HDF5 C API to build malformed inputs.

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>
#include <fstream>
#include <functional>

#include "quadriga_lib.hpp"

static bool hdf_test_file_exists(const std::string &fn)
{
    std::ifstream f(fn);
    return f.good();
}

// A valid 1-snapshot channel with 4 RX, 2 TX and 50 paths
static quadriga_lib::channel<double> hdf_test_channel(arma::uword n_path = 50)
{
    quadriga_lib::channel<double> c;
    c.name = "test";
    c.center_frequency = {3e9};
    c.tx_pos = arma::mat(3, 1, arma::fill::ones);
    c.rx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.coeff_re.push_back(arma::cube(4, 2, n_path, arma::fill::ones));
    c.coeff_im.push_back(arma::cube(4, 2, n_path, arma::fill::ones));
    c.delay.push_back(arma::cube(4, 2, n_path, arma::fill::ones));
    return c;
}

// Names of the channel groups reachable from the root group
static std::vector<std::string> hdf_test_channel_groups(const std::string &fn)
{
    std::vector<std::string> names;
    hid_t f = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    H5G_info_t info;
    H5Gget_info(f, &info);
    for (hsize_t i = 0; i < info.nlinks; ++i)
    {
        char buf[64] = {0};
        H5Lget_name_by_idx(f, ".", H5_INDEX_NAME, H5_ITER_INC, i, buf, sizeof(buf), H5P_DEFAULT);
        if (std::string(buf).rfind("channel_", 0) == 0)
            names.push_back(buf);
    }
    H5Fclose(f);
    return names;
}

// Replace an attribute by a scalar unsigned value
static void hdf_test_set_uint_attr(hid_t loc, const char *name, unsigned value)
{
    H5E_BEGIN_TRY { H5Adelete(loc, name); } H5E_END_TRY; // May not exist
    hid_t s = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_UINT, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_UINT, &value);
    H5Aclose(a), H5Sclose(s);
}

// Replace a dataset by a float dataset of the given shape (written with zeros if 'write' is true)
static void hdf_test_replace_dset(hid_t loc, const char *name, int rank, const hsize_t *dims, bool write = true)
{
    H5E_BEGIN_TRY { H5Ldelete(loc, name, H5P_DEFAULT); } H5E_END_TRY; // May not exist
    hid_t s = H5Screate_simple(rank, dims, nullptr);
    hid_t d = H5Dcreate2(loc, name, H5T_NATIVE_FLOAT, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (write)
    {
        hssize_t n = H5Sget_simple_extent_npoints(s);
        std::vector<float> zeros((size_t)n, 0.0f);
        H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, zeros.data());
    }
    H5Dclose(d), H5Sclose(s);
}

TEST_CASE("HDF - Slot rewrite replaces the slot completely")
{
    const std::string fn = "test_hdf_rewrite.hdf5";

    // Case 1: a dataset written with hdf5_write_dset, then a channel with par "b" into the same slot
    std::remove(fn.c_str());
    std::any a = 1.0;
    quadriga_lib::hdf5_write_dset(fn, "a", &a);
    auto c = hdf_test_channel(3);
    c.par_names = {"b"};
    c.par_data = {std::any(2.0)};
    CHECK(quadriga_lib::hdf5_write(&c, fn) == 1);
    {
        auto r = quadriga_lib::hdf5_read_channel<double>(fn);
        REQUIRE(r.par_names.size() == 1);
        CHECK(r.par_names[0] == "b");
        CHECK(std::any_cast<double>(r.par_data[0]) == 2.0);
        CHECK(r.coeff_re.size() == 1);
        CHECK(!quadriga_lib::hdf5_read_dset(fn, "a").has_value());
        std::vector<std::string> names;
        CHECK(quadriga_lib::hdf5_read_dset_names(fn, &names) == 1);
        REQUIRE(names.size() == 1);
        CHECK(names[0] == "b");
    }

    // Case 2: a parameter-only channel written twice to the same slot
    std::remove(fn.c_str());
    quadriga_lib::channel<double> p;
    p.par_names = {"x"};
    p.par_data = {std::any(1.0)};
    CHECK(quadriga_lib::hdf5_write(&p, fn) == 0);
    p.par_data = {std::any(99.0)};
    CHECK(quadriga_lib::hdf5_write(&p, fn) == 1);
    {
        auto r = quadriga_lib::hdf5_read_channel<double>(fn);
        REQUIRE(r.par_names.size() == 1);
        CHECK(std::any_cast<double>(r.par_data[0]) == 99.0);
    }

    // Five rewrites of one slot leave exactly one channel group of that slot; other slots are kept
    std::remove(fn.c_str());
    quadriga_lib::hdf5_create(fn, 4);
    auto other = hdf_test_channel(2);
    other.name = "other";
    quadriga_lib::hdf5_write(&other, fn, 1);
    for (int i = 0; i < 5; ++i)
    {
        c.name = "rewrite " + std::to_string(i);
        CHECK(quadriga_lib::hdf5_write(&c, fn, 0) == (i == 0 ? 0 : 1));
    }
    CHECK(hdf_test_channel_groups(fn).size() == 2);
    CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 0).name == "rewrite 4");
    CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 1).name == "other");

    // hdf5_write_dset adds a dataset and keeps the rest of the slot; an existing name is not overwritten
    std::any v = arma::vec{1.0, 2.0};
    quadriga_lib::hdf5_write_dset(fn, "extra", &v, 1);
    {
        auto r = quadriga_lib::hdf5_read_channel<double>(fn, 1);
        CHECK(r.name == "other");
        CHECK(r.coeff_re.size() == 1);
        CHECK(r.delay.size() == 1);
        REQUIRE(r.par_names.size() == 1);
        CHECK(r.par_names[0] == "extra");
    }
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(fn, "extra", &v, 1), std::invalid_argument);
    CHECK(hdf_test_channel_groups(fn).size() == 2);

    // An invalid dataset name does not reserve the empty slot
    std::any w = 1.0;
    CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(fn, "bad name", &w, 2), std::invalid_argument);
    arma::u32_vec id;
    quadriga_lib::hdf5_read_layout(fn, &id);
    CHECK(id(2) == 0);

    std::remove(fn.c_str());
}

TEST_CASE("HDF - Malformed files throw instead of overflowing buffers")
{
    const std::string fn = "test_hdf_malformed.hdf5";
    auto patch = [&](const std::function<void(hid_t)> &f)
    {
        std::remove(fn.c_str());
        auto c = hdf_test_channel();
        quadriga_lib::hdf5_write(&c, fn); // 1 snapshot: the data sits in /channel_1
        hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t g = H5Gopen2(file, "/channel_1", H5P_DEFAULT);
        f(g);
        H5Gclose(g), H5Fclose(file);
    };

    // NumPath 50 -> 1
    patch([](hid_t g) { hdf_test_set_uint_attr(g, "NumPath", 1); });
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<float>(fn), std::invalid_argument);

    // NumRx 4 -> 1
    patch([](hid_t g) { hdf_test_set_uint_attr(g, "NumRx", 1); });
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn), std::invalid_argument);

    // coeff_im missing
    patch([](hid_t g) { H5Ldelete(g, "coeff_im", H5P_DEFAULT); });
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn), std::invalid_argument);

    // delay of rank 1 and of rank 4
    patch([](hid_t g) { hsize_t d[1] = {400}; hdf_test_replace_dset(g, "delay", 1, d); });
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn), std::invalid_argument);
    patch([](hid_t g) { hsize_t d[4] = {50, 2, 4, 2}; hdf_test_replace_dset(g, "delay", 4, d); });
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn), std::invalid_argument);

    // Name as a 300-byte string
    patch([](hid_t g)
          {
              H5Adelete(g, "Name");
              std::string nm(300, 'x');
              hid_t t = H5Tcopy(H5T_C_S1);
              H5Tset_size(t, 300);
              hid_t s = H5Screate(H5S_SCALAR), a = H5Acreate2(g, "Name", t, s, H5P_DEFAULT, H5P_DEFAULT);
              H5Awrite(a, t, nm.c_str());
              H5Aclose(a), H5Sclose(s), H5Tclose(t); });
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn), std::invalid_argument);

    // Huge extents that match the attributes, but no data in the file: rejected before any allocation (64 GB)
    patch([](hid_t g)
          {
              hdf_test_set_uint_attr(g, "NumPath", 1u << 30);
              hsize_t d[3] = {1ULL << 30, 2, 4};
              hdf_test_replace_dset(g, "coeff_re", 3, d, false);
              hdf_test_replace_dset(g, "coeff_im", 3, d, false);
              hdf_test_replace_dset(g, "delay", 3, d, false); });
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn), std::invalid_argument);

    // A huge snapshot count with one snapshot group (old files pad missing snapshots with empty entries)
    patch([](hid_t g)
          {
              hdf_test_set_uint_attr(g, "NumSnap", 4000000000u);
              hid_t s = H5Gcreate2(g, "Snap_0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
              hdf_test_set_uint_attr(s, "NumPath", 1);
              hsize_t d[1] = {1};
              hdf_test_replace_dset(s, "path_gain", 1, d);
              H5Gclose(s); });
    CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn), std::invalid_argument);

    // The unpatched file reads back
    patch([](hid_t) {});
    auto r = quadriga_lib::hdf5_read_channel<double>(fn);
    CHECK(r.is_valid().empty());
    CHECK(r.coeff_im[0].n_slices == 50);
    CHECK(arma::all(arma::vectorise(r.coeff_im[0]) == 1.0));

    // Unstructured dataset of rank 4 is not supported: empty std::any, no exception
    {
        std::any x = 1.0;
        quadriga_lib::hdf5_write_dset(fn, "x", &x);
        hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t g = H5Gopen2(file, "/channel_1/Data", H5P_DEFAULT);
        hsize_t d[4] = {2, 2, 2, 2};
        hdf_test_replace_dset(g, "par_r4", 4, d);
        H5Gclose(g), H5Fclose(file);
        CHECK(!quadriga_lib::hdf5_read_dset(fn, "r4").has_value());
        auto r4 = quadriga_lib::hdf5_read_channel<double>(fn);
        CHECK(r4.par_names.size() == 1); // only "x"
    }

    std::remove(fn.c_str());
}

TEST_CASE("HDF - Storage layout is limited to 2^32-1 slots")
{
    const std::string fn = "test_hdf_layout.hdf5";
    std::remove(fn.c_str());

    CHECK_THROWS_AS(quadriga_lib::hdf5_create(fn, 65536, 65537), std::invalid_argument);
    CHECK(!hdf_test_file_exists(fn));
    CHECK_THROWS_AS(quadriga_lib::hdf5_create(fn, 65536, 65536, 65536, 65536), std::invalid_argument);
    CHECK(!hdf_test_file_exists(fn));

    quadriga_lib::hdf5_create(fn); // 65536 slots
    CHECK_THROWS_AS(quadriga_lib::hdf5_reshape_layout(fn, 65536, 65537), std::invalid_argument);
    arma::u32_vec id;
    arma::u32_vec lay = quadriga_lib::hdf5_read_layout(fn, &id);
    CHECK(lay(0) == 65536);
    CHECK(lay(1) == 1);
    CHECK(id.n_elem == 65536);
    quadriga_lib::hdf5_reshape_layout(fn, 256, 256);
    lay = quadriga_lib::hdf5_read_layout(fn);
    CHECK(lay(0) == 256);
    CHECK(lay(1) == 256);

    // Writing outside the layout throws, inside works
    auto c = hdf_test_channel(2);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write(&c, fn, 0, 256), std::invalid_argument);
    CHECK(quadriga_lib::hdf5_write(&c, fn, 255, 255) == 0);
    CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 255, 255).coeff_re.size() == 1);

    std::remove(fn.c_str());
}

TEST_CASE("HDF - Channel name is stored without foreign memory")
{
    const std::string fn = "test_hdf_name.hdf5";
    auto raw_name = [&](std::vector<unsigned char> &raw)
    {
        hid_t f = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t a = H5Aopen_by_name(f, "/channel_1", "Name", H5P_DEFAULT, H5P_DEFAULT), t = H5Aget_type(a);
        raw.assign(H5Tget_size(t), 0xFF);
        H5Aread(a, t, raw.data());
        H5Tclose(t), H5Aclose(a), H5Fclose(f);
    };

    for (size_t len : {0, 20, 255})
    {
        std::remove(fn.c_str());
        auto c = hdf_test_channel(2);
        c.name = std::string(len, 'N');
        quadriga_lib::hdf5_write(&c, fn);
        std::vector<unsigned char> raw;
        raw_name(raw);
        REQUIRE(raw.size() == 256);
        size_t non_zero = 0;
        for (size_t i = len; i < raw.size(); ++i)
            non_zero += raw[i] != 0;
        CHECK(non_zero == 0);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn).name == c.name);
    }

    // A name of 300 characters is rejected by the validation; with assume_valid it is stored truncated to 255
    std::remove(fn.c_str());
    auto c = hdf_test_channel(2);
    c.name = std::string(300, 'L');
    CHECK_THROWS_AS(quadriga_lib::hdf5_write(&c, fn), std::invalid_argument);
    quadriga_lib::hdf5_write(&c, fn, 0, 0, 0, 0, true);
    CHECK(quadriga_lib::hdf5_read_channel<double>(fn).name == std::string(255, 'L'));

    std::remove(fn.c_str());
}

TEST_CASE("HDF - Round trip keeps sizes and types")
{
    const std::string fn = "test_hdf_roundtrip.hdf5";

    auto make = [](arma::uword n_snap, arma::uvec n_path)
    {
        quadriga_lib::channel<double> c;
        c.name = "rt";
        c.center_frequency = {3e9};
        c.tx_pos = arma::mat(3, 1, arma::fill::randu);
        c.rx_pos = arma::mat(3, n_snap, arma::fill::randu);
        for (arma::uword s = 0; s < n_snap; ++s)
        {
            arma::uword np = n_path(s);
            c.coeff_re.push_back(arma::cube(2, 3, np, arma::fill::randu));
            c.coeff_im.push_back(arma::cube(2, 3, np, arma::fill::randu));
            c.delay.push_back(arma::cube(1, 1, np, arma::fill::randu) * 1e-6);
            c.path_gain.push_back(arma::vec(np, arma::fill::randu));
            c.path_angles.push_back(arma::mat(np, 4, arma::fill::randu));
            c.no_interact.push_back(arma::u32_vec(np, arma::fill::ones));
            c.interact_coord.push_back(arma::mat(3, np, arma::fill::randu));
        }
        return c;
    };
    auto check_same = [&](const quadriga_lib::channel<double> &c)
    {
        std::remove(fn.c_str());
        quadriga_lib::hdf5_write(&c, fn);
        auto r = quadriga_lib::hdf5_read_channel<double>(fn);
        CHECK(r.is_valid().empty());
        CHECK(r.n_snap() == c.n_snap());
        CHECK(arma::size(r.center_frequency) == arma::size(c.center_frequency));
        REQUIRE(r.coeff_re.size() == c.coeff_re.size());
        REQUIRE(r.delay.size() == c.delay.size());
        REQUIRE(r.path_gain.size() == c.path_gain.size());
        REQUIRE(r.path_angles.size() == c.path_angles.size());
        REQUIRE(r.no_interact.size() == c.no_interact.size());
        REQUIRE(r.interact_coord.size() == c.interact_coord.size());
        for (size_t s = 0; s < c.coeff_re.size(); ++s)
        {
            CHECK(arma::size(r.coeff_re[s]) == arma::size(c.coeff_re[s]));
            CHECK(arma::size(r.coeff_im[s]) == arma::size(c.coeff_im[s]));
            CHECK(arma::size(r.delay[s]) == arma::size(c.delay[s]));
            CHECK(arma::size(r.path_gain[s]) == arma::size(c.path_gain[s]));
            CHECK(arma::size(r.path_angles[s]) == arma::size(c.path_angles[s]));
            CHECK(arma::size(r.no_interact[s]) == arma::size(c.no_interact[s]));
            CHECK(arma::size(r.interact_coord[s]) == arma::size(c.interact_coord[s]));
            if (r.coeff_re[s].n_elem)
                CHECK(arma::approx_equal(r.coeff_re[s], c.coeff_re[s], "absdiff", 1e-6));
        }
    };

    check_same(make(1, {0}));       // 1 snapshot without paths
    check_same(make(2, {3, 0}));    // trailing empty snapshot, delay [1, 1, 0]
    check_same(make(3, {4, 0, 5})); // empty snapshot in the middle
    check_same(make(3, {0, 0, 2})); // leading empty snapshots
    {
        auto c = make(1, {2});
        c.center_frequency.reset(); // empty center_frequency stays empty
        check_same(c);
    }

    // Unstructured data
    std::remove(fn.c_str());
    std::vector<std::pair<std::string, std::any>> par = {
        {"s_double", 2.5}, {"vec1", arma::vec{7.0}}, {"fvec1", arma::fvec{7.0f}}, {"u32vec1", arma::Col<unsigned>{5u}},
        {"ivec1", arma::ivec{-9}}, {"row1", arma::rowvec{7.0}}, {"mat11", arma::mat(1, 1, arma::fill::ones)},
        {"cube111", arma::cube(1, 1, 1, arma::fill::ones)}, {"str_empty", std::string("")}, {"str", std::string("abc")},
        {"mat0x3", arma::mat(0, 3)}, {"cube230", arma::cube(2, 3, 0)}, {"vec0", arma::vec()}, {std::string(260, 'L'), 1.0}};
    for (auto &p : par)
        quadriga_lib::hdf5_write_dset(fn, p.first, &p.second);

    std::vector<std::string> names;
    CHECK(quadriga_lib::hdf5_read_dset_names(fn, &names) == par.size());
    REQUIRE(names.size() == par.size());
    for (size_t i = 0; i < par.size(); ++i)
    {
        CHECK(names[i] == par[i].first);
        std::any r = quadriga_lib::hdf5_read_dset(fn, par[i].first);
        unsigned long long d0[3], d1[3];
        int t0 = quadriga_lib::any_type_id(&par[i].second, d0), t1 = quadriga_lib::any_type_id(&r, d1);
        if (t0 >= 50 && t0 < 60) // arma::Row is stored as arma::Col (documented), any_type_id gives the same dims
            t0 -= 10;
        CHECK(t0 == t1);
        if (t1 != 9)
        {
            CHECK(d0[0] == d1[0]);
            CHECK(d0[1] == d1[1]);
            CHECK(d0[2] == d1[2]);
        }
    }
    CHECK(std::any_cast<std::string>(quadriga_lib::hdf5_read_dset(fn, "str_empty")).empty());
    CHECK(std::any_cast<std::string>(quadriga_lib::hdf5_read_dset(fn, "str")) == "abc");
    CHECK(std::any_cast<arma::vec>(quadriga_lib::hdf5_read_dset(fn, "vec1"))(0) == 7.0);

    auto r = quadriga_lib::hdf5_read_channel<double>(fn);
    CHECK(r.par_names.size() == par.size());

    std::remove(fn.c_str());
}

static herr_t hdf_test_error_handler(hid_t, void *) { return 0; }

TEST_CASE("HDF - No side effects on the caller's HDF5 state")
{
    const std::string fn = "test_hdf_side_effects.hdf5";
    const std::string user_fn = "test_hdf_side_effects_user.h5";
    std::remove(fn.c_str());

    // The caller's own error handler
    H5E_auto2_t old_func = nullptr;
    void *old_data = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_data);
    int marker = 0;
    H5Eset_auto2(H5E_DEFAULT, hdf_test_error_handler, &marker);

    // The caller has its own HDF5 file open
    hid_t uf = H5Fcreate(user_fn.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t n = 4;
    hid_t sp = H5Screate_simple(1, &n, nullptr);
    hid_t ds = H5Dcreate2(uf, "x", H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    quadriga_lib::hdf5_create(fn, 4);
    auto c = hdf_test_channel(2);
    quadriga_lib::hdf5_write(&c, fn, 0);
    quadriga_lib::hdf5_write(&c, fn, 0);
    quadriga_lib::hdf5_read_layout(fn);
    quadriga_lib::hdf5_read_channel<double>(fn, 0);
    std::any a = 1.0;
    quadriga_lib::hdf5_write_dset(fn, "a", &a, 1);
    quadriga_lib::hdf5_read_dset(fn, "a", 1);
    std::vector<std::string> names;
    quadriga_lib::hdf5_read_dset_names(fn, &names, 1);
    quadriga_lib::hdf5_reshape_layout(fn, 2, 2);
    CHECK_THROWS_AS(quadriga_lib::hdf5_write(&c, fn, 7), std::invalid_argument);
    {
        const std::string txt_fn = "test_hdf_side_effects_not_hdf5.txt";
        std::ofstream(txt_fn) << "This is not HDF5 data";
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_layout(txt_fn), std::invalid_argument);
        std::remove(txt_fn.c_str());
    }

    CHECK(H5Iis_valid(uf) > 0);
    CHECK(H5Iis_valid(ds) > 0);
    double v[4] = {1, 2, 3, 4};
    CHECK(H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v) >= 0);
    H5Dclose(ds), H5Sclose(sp), H5Fclose(uf);

    // The caller's error handler is restored after every call, also after a failing one
    H5E_auto2_t func = nullptr;
    void *data = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &func, &data);
    CHECK(func == hdf_test_error_handler);
    CHECK(data == &marker);
    H5Eset_auto2(H5E_DEFAULT, old_func, old_data);

    // The caller holds the same file open: the library opens and closes only its own handle
    hid_t f2 = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK_NOTHROW(quadriga_lib::hdf5_read_layout(fn));
    CHECK_NOTHROW(quadriga_lib::hdf5_read_channel<double>(fn, 0));
    CHECK(H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL) == 1);
    H5Fclose(f2);

    // A corrupted storage index throws (and leaks nothing, checked by LeakSanitizer in the sanitizer build)
    std::remove(fn.c_str());
    quadriga_lib::hdf5_create(fn, 4);
    {
        hid_t f = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t d = H5Dopen2(f, "ChannelDims", H5P_DEFAULT);
        unsigned dims[4] = {3, 1, 1, 1};
        H5Dwrite(d, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, dims);
        H5Dclose(d), H5Fclose(f);
    }
    for (int i = 0; i < 3; ++i)
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn, 0), std::invalid_argument);
    CHECK(H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL) == 0);

    std::remove(fn.c_str());
    std::remove(user_fn.c_str());
}

TEST_CASE("HDF - Strings written by other tools")
{
    const std::string fn = "test_hdf_vlstring.hdf5";
    std::remove(fn.c_str());
    std::any a = 1.0;
    quadriga_lib::hdf5_write_dset(fn, "a", &a); // creates /channel_1/Data

    const char *txt = "written by another HDF5 tool";
    hid_t f = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t g = H5Gopen2(f, "/channel_1/Data", H5P_DEFAULT);
    {
        // Variable-length string (default of h5py and MATLAB)
        hid_t t = H5Tcopy(H5T_C_S1);
        H5Tset_size(t, H5T_VARIABLE);
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t d = H5Dcreate2(g, "par_note", t, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(d, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, &txt);
        H5Dclose(d), H5Sclose(sp), H5Tclose(t);
    }
    {
        // Fixed-length NUL-padded string of 10 bytes holding "abc"
        char buf[10] = "abc";
        hid_t t = H5Tcopy(H5T_C_S1);
        H5Tset_size(t, 10);
        H5Tset_strpad(t, H5T_STR_NULLPAD);
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t d = H5Dcreate2(g, "par_padded", t, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(d, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        H5Dclose(d), H5Sclose(sp), H5Tclose(t);
    }
    H5Gclose(g), H5Fclose(f);

    std::any r = quadriga_lib::hdf5_read_dset(fn, "note");
    REQUIRE(quadriga_lib::any_type_id(&r) == 9);
    CHECK(std::any_cast<std::string>(r) == txt);
    r = quadriga_lib::hdf5_read_dset(fn, "padded");
    REQUIRE(quadriga_lib::any_type_id(&r) == 9);
    CHECK(std::any_cast<std::string>(r) == "abc");

    auto c = quadriga_lib::hdf5_read_channel<double>(fn);
    REQUIRE(c.par_names.size() == 3);
    CHECK(std::any_cast<std::string>(c.par_data[1]) == txt);

    std::remove(fn.c_str());
}
