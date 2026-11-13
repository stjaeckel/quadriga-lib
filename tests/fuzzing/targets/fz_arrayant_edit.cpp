// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Array antenna member functions in a fuzzed order on a fuzzed arrayant: is_valid, copy and move, validate, append,
// calc_beamwidth_deg, calc_directivity_dBi, combine_pattern, copy_element, export_obj_file, qdant_write / qdant_read,
// remove_zeros, set_size, reset; arrayant_rotate_pattern; and the multi-frequency functions arrayant_is_valid_multi,
// arrayant_concat_multi, arrayant_copy_element_multi, arrayant_rotate_pattern_multi, arrayant_set_element_pos_multi,
// arrayant_combine_pattern_multi, qdant_write_multi / qdant_read_multi

#include "fuzz_target.hpp"

// Up to 3 entries: copies of a with a fuzzed center frequency, or other fuzzed arrayants
template <typename T>
static std::vector<quadriga_lib::arrayant<T>> fuzzed_vector(fz::input &in, const quadriga_lib::arrayant<T> &a)
{
    std::vector<quadriga_lib::arrayant<T>> vec;
    arma::uword n = in.size(3);
    for (arma::uword f = 0; f < n; ++f)
    {
        vec.push_back(in.ConsumeIntegralInRange<int>(0, 5) == 0 ? fz::arrayant<T>(in) : a.copy());
        vec.back().center_frequency = in.value<T>();
    }
    return vec;
}

template <typename T>
static void run(fz::input &in)
{
    quadriga_lib::arrayant<T> a = fz::arrayant<T>(in);
    const std::string obj = (fz::work_dir() / "out.obj").string(), qdant = (fz::work_dir() / "out.qdant").string();

    for (int op = 0; op < 8 && in.remaining_bytes() != 0; ++op)
    {
        const arma::uword n_e = a.e_theta_re.n_slices;
        switch (in.ConsumeIntegralInRange<int>(0, 19))
        {
        case 0:
            if (in.ConsumeBool())
                fz::clean([&] { a.is_valid(); });
            else // Copy and move pass the validity mark on
                fz::clean([&] { quadriga_lib::arrayant<T> b = a; b.is_valid(); a = std::move(b); });
            break;
        case 1:
            fz::clean([&] { a.is_valid(); });
            break;
        case 2:
        {
            quadriga_lib::arrayant<T> b = in.ConsumeBool() ? a.copy() : fz::arrayant<T>(in);
            if (n_e <= 16) // keeps repeated appends small
                fz::clean([&] { a = quadriga_lib::arrayant_concat(a, b); });
            break;
        }
        case 3:
        {
            T bw_az, bw_el, p_az, p_el;
            arma::uword i = in.index(n_e);
            T threshold = in.value<T>();
            bool all = in.ConsumeBool();
            fz::clean([&] { a.calc_beamwidth_deg(i, threshold, &bw_az, all ? &bw_el : nullptr, all ? &p_az : nullptr, &p_el); });
            break;
        }
        case 4:
        {
            arma::uword i = in.index(n_e);
            fz::clean([&] { a.calc_directivity_dBi(i); });
            break;
        }
        case 5:
        {
            arma::Col<T> az = in.col<T>(in.size(16)), el = in.col<T>(in.size(16));
            if (in.ConsumeBool())
                az = arma::sort(az), el = arma::sort(el);
            const int mode = in.ConsumeIntegralInRange<int>(0, 3);
            T speed = in.ConsumeBool() ? T(299792458.0) : in.value<T>();
            fz::clean([&] { a = quadriga_lib::arrayant_combine_pattern(a, mode & 1 ? &az : nullptr, mode & 2 ? &el : nullptr, speed); });
            break;
        }
        case 6:
        {
            arma::uword src = in.index(n_e), dst = in.index(n_e);
            fz::clean([&] { quadriga_lib::arrayant_copy_element(a, src, {dst}); });
            break;
        }
        case 7:
        {
            arma::uword src = in.index(n_e);
            arma::uvec dst(in.size(4));
            for (auto &d : dst)
                d = in.index(n_e);
            fz::clean([&] { quadriga_lib::arrayant_copy_element(a, src, dst); });
            break;
        }
        case 8:
        {
            T range = in.value<T>(), radius = in.value<T>();
            arma::uword n_div = in.size(3);
            arma::uvec i_el(in.size(3));
            for (auto &i : i_el)
                i = in.index(n_e);
            const char *maps[] = {"jet", "parula", "winter", "hot", "turbo", "copper", "spring", "cool", "gray", "autumn", "summer", ""};
            std::string cmap = in.ConsumeBool() ? maps[in.ConsumeIntegralInRange<int>(0, 11)] : in.ConsumeRandomLengthString(12);
            fz::clean([&] { quadriga_lib::arrayant_export_obj_file(obj, a, range, cmap, radius, n_div, &i_el); });
            break;
        }
        case 9:
        {
            unsigned id = in.ConsumeIntegralInRange<unsigned>(0, 3);
            arma::u32_mat layout(in.size(3), in.size(3));
            for (auto &l : layout)
                l = in.ConsumeIntegralInRange<unsigned>(0, 4);
            if (in.ConsumeBool())
                std::filesystem::remove(qdant);
            fz::clean([&] { quadriga_lib::qdant_write(qdant, a, id, &layout); });
            fz::clean([&] { arma::u32_mat lay; quadriga_lib::qdant_read<T>(qdant, in.ConsumeIntegralInRange<unsigned>(0, 4), &lay); });
            break;
        }
        case 10:
            if (in.ConsumeBool())
                fz::clean([&] { quadriga_lib::arrayant_remove_zeros(a); });
            else
                fz::clean([&] { quadriga_lib::arrayant<T> out = a.copy(); quadriga_lib::arrayant_remove_zeros(out); a = out; });
            break;
        case 11:
        {
            const arma::Col<T> orientation = {in.value<T>(), in.value<T>(), in.value<T>()};
            unsigned usage = in.ConsumeIntegralInRange<unsigned>(0, 5);
            arma::uvec i_el(in.size(3));
            for (auto &i : i_el)
                i = in.index(n_e);
            fz::clean([&] { quadriga_lib::arrayant_rotate_pattern(a, orientation, usage, &i_el); });
            break;
        }
        case 12:
        {
            arma::uword ne = in.size(8), na = in.size(12), nel = in.size(6), np = in.size(6);
            fz::clean([&] { a.set_size(ne, na, nel, np); });
            // set_size leaves the patterns and grids undefined; a caller fills them
            a.azimuth_grid = arma::linspace<arma::Col<T>>(T(-3.14159265), T(3.14159265), a.e_theta_re.n_cols);
            a.elevation_grid = arma::linspace<arma::Col<T>>(T(-1.5707963), T(1.5707963), a.e_theta_re.n_rows);
            for (arma::Cube<T> *c : {&a.e_theta_re, &a.e_theta_im, &a.e_phi_re, &a.e_phi_im})
                *c = in.cube<T>(c->n_rows, c->n_cols, c->n_slices);
            break;
        }
        case 13:
            if (in.ConsumeIntegralInRange<int>(0, 3) == 0)
                fz::clean([&] { a.reset(); });
            else
                a = fz::arrayant<T>(in);
            break;
        case 14:
        {
            auto vec = fuzzed_vector<T>(in, a);
            fz::clean([&] { quadriga_lib::arrayant_is_valid_multi(vec); });
            auto vec2 = in.ConsumeBool() ? vec : fuzzed_vector<T>(in, a);
            fz::clean([&] { quadriga_lib::arrayant_concat_multi(vec, vec2); });
            break;
        }
        case 15:
        {
            auto vec = fuzzed_vector<T>(in, a);
            arma::uword src = in.index(n_e);
            arma::uvec dst(in.size(4));
            for (auto &d : dst)
                d = in.index(n_e);
            fz::clean([&] { quadriga_lib::arrayant_copy_element_multi(vec, src, dst); });
            fz::clean([&] { quadriga_lib::arrayant_copy_element_multi(vec, in.index(n_e), arma::uvec{in.index(n_e)}); });
            break;
        }
        case 16:
        {
            auto vec = fuzzed_vector<T>(in, a);
            const arma::Col<T> orientation = {in.value<T>(), in.value<T>(), in.value<T>()};
            unsigned usage = in.ConsumeIntegralInRange<unsigned>(0, 5);
            arma::uvec i_el(in.size(3));
            for (auto &i : i_el)
                i = in.index(n_e);
            fz::clean([&] { quadriga_lib::arrayant_rotate_pattern_multi(vec, orientation, usage, &i_el); });
            break;
        }
        case 17:
        {
            auto vec = fuzzed_vector<T>(in, a);
            arma::uvec i_el(in.size(4));
            for (auto &i : i_el)
                i = in.index(n_e);
            arma::Mat<T> pos = in.ConsumeBool() ? in.mat<T>(3, i_el.is_empty() ? n_e : i_el.n_elem)
                                                : in.mat<T>(in.size(4), in.size(7));
            fz::clean([&] { quadriga_lib::arrayant_set_element_pos_multi(vec, pos, i_el); });
            break;
        }
        case 18:
        {
            auto vec = fuzzed_vector<T>(in, a);
            arma::Col<T> az = in.col<T>(in.size(16)), el = in.col<T>(in.size(16)), fr = in.col<T>(in.size(4));
            const int mode = in.ConsumeIntegralInRange<int>(0, 7);
            T speed = in.ConsumeBool() ? T(343.0) : in.value<T>();
            fz::clean([&] { quadriga_lib::arrayant_combine_pattern_multi(vec, mode & 1 ? &az : nullptr, mode & 2 ? &el : nullptr,
                                                                         mode & 4 ? &fr : nullptr, speed); });
            break;
        }
        default:
        {
            auto vec = fuzzed_vector<T>(in, a);
            fz::clean([&] { quadriga_lib::qdant_write_multi(qdant, vec); });
            fz::clean([&] { arma::u32_mat lay; quadriga_lib::qdant_read_multi<T>(qdant, &lay); });
        }
        }
    }
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
