// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// HDF5 functions on files that were changed with the HDF5 C library: malformed files (the reader throws instead of
// allocating or overflowing buffers), strings written by other tools, and no side effects on the HDF5 state of a
// caller that uses the HDF5 C library itself. The behavior on files written by the library is tested in
// test_hdf_functions.cpp, the file structure in test_hdf5_interop.cpp. get_HDF5_version is checked here as well.

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>
#include <fstream>
#include <functional>

#include "quadriga_lib.hpp"

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

TEST_CASE("hdf5_read_channel - malformed files throw instead of overflowing buffers")
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

static herr_t hdf_test_error_handler(hid_t, void *) { return 0; }

TEST_CASE("hdf5 - no side effects on the caller's HDF5 state")
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

TEST_CASE("hdf5_read_dset - strings written by other tools")
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

TEST_CASE("get_HDF5_version - returns the version of the HDF5 headers as major.minor.release")
{
    const std::string expected = std::to_string(H5_VERS_MAJOR) + "." + std::to_string(H5_VERS_MINOR) + "." +
                                 std::to_string(H5_VERS_RELEASE);
    CHECK(quadriga_lib::get_HDF5_version() == expected);
}
