// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Channel objects with fuzzed members (consistent or mismatched sizes, unstructured par_names / par_data of all
// types): channel::is_valid, calc_effective_path_gain, add_paths, write_paths_to_obj_file, then hdf5_create,
// hdf5_write, hdf5_write_dset, hdf5_reshape_layout and the readers hdf5_read_layout, hdf5_read_channel,
// hdf5_read_dset_names, hdf5_read_dset

#include "fuzz_target.hpp"

// An unstructured value of one of the supported types, or an empty or unsupported one
template <typename T>
static std::any fuzzed_any(fz::input &in)
{
    switch (in.ConsumeIntegralInRange<int>(0, 17))
    {
    case 0:
        return in.ConsumeRandomLengthString(16);
    case 1:
        return in.value<float>();
    case 2:
        return in.value<double>();
    case 3:
        return in.ConsumeIntegral<int>();
    case 4:
        return in.ConsumeIntegral<long long>();
    case 5:
        return in.ConsumeIntegral<unsigned>();
    case 6:
        return in.ConsumeIntegral<unsigned long long>();
    case 7:
        return in.col<T>(in.size(6));
    case 8:
        return in.mat<T>(in.size(4), in.size(4));
    case 9:
        return in.cube<T>(in.size(3), in.size(3), in.size(3));
    case 10:
        return arma::Row<T>(in.col<T>(in.size(5)).t());
    case 11:
        return arma::Mat<int>(in.size(3), in.size(3), arma::fill::value(in.ConsumeIntegral<int>()));
    case 12:
        return arma::Cube<unsigned>(in.size(3), in.size(3), in.size(3), arma::fill::value(in.ConsumeIntegral<unsigned>()));
    case 13:
        return arma::Col<arma::uword>(in.size(4), arma::fill::value(in.ConsumeIntegral<arma::uword>()));
    case 14:
        return arma::Mat<arma::sword>(in.size(3), in.size(3), arma::fill::value(in.ConsumeIntegral<arma::sword>()));
    case 15:
        return std::vector<int>(in.size(3), 1); // unsupported type
    case 16:
        return std::any();
    default:
        return in.ConsumeBool();
    }
}

static std::string fuzzed_name(fz::input &in)
{
    const char *names[] = {"x", "par_a", "a_b_1", "name", "", "a b", "a/b", "."};
    return in.ConsumeBool() ? names[in.ConsumeIntegralInRange<int>(0, 7)] : in.ConsumeRandomLengthString(12);
}

template <typename T>
static void run(fz::input &in)
{
    quadriga_lib::channel<T> c;
    const bool consistent = in.ConsumeIntegralInRange<int>(0, 3) != 0;
    auto sz = [&](arma::uword n) { return consistent ? n : in.size(n + 1); };
    const arma::uword n_snap = in.size(3), n_rx = in.size(3), n_tx = in.size(3);
    const uint32_t mask = in.ConsumeIntegral<uint32_t>(); // which members are present

    if (in.ConsumeBool())
        c.name = in.ConsumeRandomLengthString(16);
    auto pos = [&](bool optional)
    {
        switch (in.ConsumeIntegralInRange<int>(0, 3))
        {
        case 0:
            return optional ? arma::Mat<T>() : in.mat<T>(3, 1);
        case 1:
            return in.mat<T>(3, 1);
        case 2:
            return in.mat<T>(3, n_snap);
        default:
            return in.mat<T>(in.size(4), in.size(4));
        }
    };
    c.tx_pos = pos(false), c.rx_pos = pos(false), c.tx_orientation = pos(true), c.rx_orientation = pos(true);
    if (mask & 1)
        c.center_frequency = in.col<T>(in.ConsumeBool() ? 1 : sz(n_snap));

    for (arma::uword s = 0; s < n_snap; ++s)
    {
        const arma::uword n_path = in.size(4);
        if (mask & 2)
            c.coeff_re.push_back(in.cube<T>(n_rx, n_tx, n_path)), c.coeff_im.push_back(in.cube<T>(sz(n_rx), sz(n_tx), sz(n_path)));
        if (mask & 4)
            c.delay.push_back(in.ConsumeBool() ? in.cube<T>(1, 1, sz(n_path)) : in.cube<T>(sz(n_rx), sz(n_tx), sz(n_path)));
        if (mask & 8)
            c.path_gain.push_back(in.col<T>(sz(n_path)));
        if (mask & 16)
            c.path_length.push_back(in.col<T>(sz(n_path)));
        if (mask & 32)
            c.path_polarization.push_back(in.mat<T>(sz(8), sz(n_path)));
        if (mask & 64)
            c.path_angles.push_back(in.mat<T>(sz(n_path), sz(4)));
        if (mask & 128)
            c.path_fbs_pos.push_back(in.mat<T>(sz(3), sz(n_path))), c.path_lbs_pos.push_back(in.mat<T>(sz(3), sz(n_path)));
        if (mask & 256)
        {
            arma::Col<unsigned> no_int(sz(n_path));
            for (auto &v : no_int)
                v = in.ConsumeIntegralInRange<int>(0, 15) == 0 ? in.ConsumeIntegral<unsigned>() : in.ConsumeIntegralInRange<unsigned>(0, 3);
            arma::uword total = 0;
            for (auto v : no_int)
                total += v < 4 ? v : 0;
            c.no_interact.push_back(no_int);
            c.interact_coord.push_back(in.mat<T>(sz(3), sz(total)));
        }
    }
    if (!consistent && in.ConsumeBool()) // a member vector of another length
    {
        const arma::uword n = in.size(n_snap + 1);
        switch (in.ConsumeIntegralInRange<int>(0, 3))
        {
        case 0:
            c.coeff_im.resize(n);
            break;
        case 1:
            c.delay.resize(n);
            break;
        case 2:
            c.no_interact.resize(n);
            break;
        default:
            c.path_gain.resize(n);
        }
    }

    const arma::uword n_par = in.size(4);
    for (arma::uword i = 0; i < n_par; ++i)
        c.par_names.push_back(fuzzed_name(in)), c.par_data.push_back(fuzzed_any<T>(in));
    if (!consistent && in.ConsumeBool())
        c.par_data.resize(in.size(n_par + 1));
    c.initial_position = in.ConsumeBool() ? 0 : in.ConsumeIntegral<int>();

    // The integrity check may be skipped only for an object that passed it
    std::string err = "?";
    fz::clean([&] { err = c.is_valid(); });
    const bool valid = err.empty(), skip_check = valid && in.ConsumeBool();

    fz::clean([&] { c.calc_effective_path_gain(skip_check); });
    {
        const arma::uword i_snap = in.index(n_snap), n_add = in.size(4);
        const uint16_t add = in.ConsumeIntegral<uint16_t>();
        arma::Cube<T> a_re = in.cube<T>(n_rx, n_tx, n_add), a_im = in.cube<T>(n_rx, n_tx, sz(n_add)),
                      a_dl = in.ConsumeBool() ? in.cube<T>(1, 1, n_add) : in.cube<T>(sz(n_rx), sz(n_tx), n_add);
        arma::u32_vec a_ni(sz(n_add));
        for (auto &v : a_ni)
            v = in.ConsumeIntegralInRange<unsigned>(0, 3);
        arma::Mat<T> a_ic = in.mat<T>(sz(3), sz((arma::uword)arma::accu(a_ni)));
        arma::Col<T> a_pg = in.col<T>(sz(n_add)), a_pl = in.col<T>(sz(n_add));
        arma::Mat<T> a_pp = in.mat<T>(sz(8), sz(n_add)), a_pa = in.mat<T>(sz(n_add), sz(4)), a_fbs = in.mat<T>(sz(3), sz(n_add)),
                     a_lbs = in.mat<T>(sz(3), sz(n_add));
        auto opt = [&](auto &x, int bit) { return (add >> bit) & 1 ? &x : nullptr; };
        fz::clean([&] { c.add_paths(i_snap, opt(a_re, 0), opt(a_im, 1), opt(a_dl, 2), opt(a_ni, 3), opt(a_ic, 4), opt(a_pg, 5),
                                    opt(a_pl, 6), opt(a_pp, 7), opt(a_pa, 8), opt(a_fbs, 9), opt(a_lbs, 10)); });
    }
    {
        const std::string obj = (fz::work_dir() / "paths.obj").string();
        const arma::uword max_paths = in.size(5), n_edges = in.size(6);
        T g_max = in.value<T>(), g_min = in.value<T>(), r_max = in.value<T>(), r_min = in.value<T>();
        arma::uvec i_snap(in.size(3));
        for (auto &i : i_snap)
            i = in.index(c.n_snap());
        fz::clean([&] { c.write_paths_to_obj_file(obj, max_paths, g_max, g_min, "jet", i_snap, r_max, r_min, n_edges); });
    }

    // HDF5: a new file, written at a fuzzed index (inside or outside the layout), then read back
    const std::string fn = (fz::work_dir() / "ch.hdf5").string();
    std::filesystem::remove(fn);
    if (in.ConsumeBool())
    {
        unsigned nx = (unsigned)in.size(3), ny = (unsigned)in.size(3), nz = (unsigned)in.size(2), nw = (unsigned)in.size(2);
        fz::clean([&] { quadriga_lib::hdf5_create(fn, nx, ny, nz, nw); });
    }
    unsigned ix = (unsigned)in.index(2), iy = (unsigned)in.index(2), iz = (unsigned)in.size(1), iw = (unsigned)in.size(1);
    fz::clean([&] { quadriga_lib::hdf5_write(&c, fn, ix, iy, iz, iw, skip_check); });
    if (in.ConsumeBool()) // overwrite the slot
        fz::clean([&] { quadriga_lib::hdf5_write(&c, fn, ix, iy, iz, iw, false); });
    {
        std::string name = fuzzed_name(in);
        std::any par = fuzzed_any<T>(in);
        std::string prefix = in.ConsumeBool() ? "par_" : in.ConsumeRandomLengthString(6);
        fz::clean([&] { quadriga_lib::hdf5_write_dset(fn, name, &par, ix, iy, iz, iw, prefix); });
    }
    if (in.ConsumeBool())
    {
        unsigned nx = (unsigned)in.size(4), ny = (unsigned)in.size(4), nz = (unsigned)in.size(2), nw = (unsigned)in.size(2);
        fz::clean([&] { quadriga_lib::hdf5_reshape_layout(fn, nx, ny, nz, nw); });
    }
    fz::clean([&] { arma::u32_vec id; quadriga_lib::hdf5_read_layout(fn, &id); });
    fz::clean([&] { quadriga_lib::hdf5_read_channel<T>(fn, ix, iy, iz, iw); });
    fz::clean([&] { quadriga_lib::hdf5_read_channel<double>(fn, 0); });
    std::vector<std::string> names;
    fz::clean([&] { quadriga_lib::hdf5_read_dset_names(fn, &names, ix, iy, iz, iw); });
    for (const auto &n : names)
        fz::clean([&] { quadriga_lib::hdf5_read_dset(fn, n, ix, iy, iz, iw); });
    fz::clean([&] { quadriga_lib::hdf5_read_dset(fn, fuzzed_name(in), ix, iy, iz, iw); });
}

void fuzz_seeds(const fz::seeds &) {}

void fuzz_one(const uint8_t *data, size_t size)
{
    fz::input in(data, size);
    if (in.ConsumeBool())
        run<float>(in);
    else
        run<double>(in);
}
