// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Scalar options of the function templates are declared as double, so dtype is deduced from the data arguments only
// and integer or double literals work with float data. The calls in calls_with_literals() are compiled and linked,
// but not run (the branch is never taken); a few cheap calls check that the values arrive.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

using T = float;

// Float data with int and double literals for every scalar option; compiled and linked only
static void calls_with_literals()
{
    quadriga_lib::arrayant<T> a;
    std::vector<quadriga_lib::arrayant<T>> av;
    quadriga_lib::channel<T> c;
    arma::Mat<T> m, m2, v;
    arma::Col<T> col;
    arma::Cube<T> cu, cu2, cu3, cu4;
    arma::umat faces;
    arma::uvec ui;
    std::vector<arma::Cube<T>> vc, vc2, vc3, vc4, vc5, vc6;
    std::vector<arma::Col<T>> vcol;
    std::vector<arma::Mat<T>> vm;
    std::unordered_map<std::string, std::vector<T>> prop;
    arma::u32_vec u32;
    arma::Col<short> sc;
    std::vector<uint8_t> u8;

    // quadriga_arrayant.hpp
    quadriga_lib::calc_beamwidth(a, ui, 3);
    quadriga_lib::apply_port_weights(a, col, col, 343);
    quadriga_lib::export_pattern_obj("a.obj", a, 30.0, "jet", 1.0);
    quadriga_lib::calc_beamwidth(av, ui, 3);
    quadriga_lib::apply_port_weights(av, col, col, col, 343);
    a = quadriga_lib::generate_antenna<T>("omni", 10);
    a = quadriga_lib::generate_antenna<T>("custom", 10, NAN, {90, 90.0});
    a = quadriga_lib::generate_antenna("ula", 10, 3.5e9, {}, {1, 4}, {0.5}, {}, {}, {}, {}, {}, {}, a);
    a = quadriga_lib::generate_antenna<T>("multibeam", 10, 3.5e9, {120, 120}, {2, 2}, {0.5}, {1});
    a = quadriga_lib::generate_antenna<T>("3GPP", 10, 3.5e9, {}, {2, 2}, {0.5}, {3, 2}, {1}, {6, 0});
    av = quadriga_lib::generate_speaker<T>("piston", 0.05, 80, 12000, 12, 12, 85, "hemisphere", 90, 60, 1000, 0.15, 0.25, col, 5);
    quadriga_lib::arrayant<double> ad = quadriga_lib::generate_antenna("omni", 10); // dtype defaults to double

    // quadriga_channel.hpp
    quadriga_lib::channel_export_obj_file("a.obj", c, 0, -60, -140, "jet", ui, 0.05, 0.01);
    quadriga_lib::quantize_delays(vc, vc, vc, vc2, vc3, vc4, 5e-9, 48, 1);
    quadriga_lib::get_channels_spherical(a, a, m, m, m, m, m, m, col, col, m, cu, cu2, cu3, 3.5e9);
    quadriga_lib::get_channels_spherical_vec(a, a, m, m, m, m, vm, vm, vcol, vcol, vm, vc, vc2, vc3, 3.5e9);
    quadriga_lib::get_channels_multifreq(av, av, m, m, m, m, m, m, m, col, cu, col, col, vc, vc2, vc3, false, false, 343);
    quadriga_lib::get_channels_irs(a, a, a, m, m, m, m, m, m, m, m, col, col, m, m, m, col, col, m, cu, cu2, cu3, 0, -140, 3.5e9);
    quadriga_lib::get_channels_planar(a, a, m, m, m, m, col, col, col, col, col, col, m, cu, cu2, cu3, 3.5e9);
    quadriga_lib::get_channels_planar_vec(a, a, m, m, m, m, vcol, vcol, vcol, vcol, vcol, vcol, vm, vc, vc2, vc3, 3.5e9);

    // quadriga_sos.hpp
    m = quadriga_lib::sos_preset<T>("exponential", &col, &col, 10);
    quadriga_lib::sos_val(m, m, m, v, m, m, col, col, 1);
    quadriga_lib::sos_remove_bias(m, m2, m, m, &v, col, col, &m, &m, 1);
    m = quadriga_lib::sos_generate(col, col, 256, &col, &m, 3, true, 10, true, 0, 1.0);

    // quadriga_tools.hpp
    col = quadriga_lib::calc_delay_spread(vcol, vcol, 100, 0);
    quadriga_lib::calc_angular_spread(vcol, vcol, vcol, &col, &col, &m, &vcol, &vcol, false, false, 1);
    quadriga_lib::calc_rician_k_factor(vcol, vcol, m, m, &col, &col, 0.01);
    quadriga_lib::calc_cross_polarization_ratio(vcol, vm, vcol, m, m, &m2, &col, false, 0.01);
    quadriga_lib::calc_diffraction_gain(m, m, m, ui, prop, 3.5e9);
    quadriga_lib::generate_diffraction_paths(m, m, 3.5e9, 2, cu, cu2, cu3, cu4);
    quadriga_lib::icosphere(2, 1, m);
    quadriga_lib::obj_file_write("a.obj", m, ui, ui, {}, {}, &v, &faces, m2, faces, m2, 0.001);
    ui = quadriga_lib::obj_overlap_test(m, ui, nullptr, 0);
    quadriga_lib::path_to_tube(m, v, faces, 0.1, 3);
    quadriga_lib::point_inside_mesh(m, m, ui, 1);
    quadriga_lib::ray_mesh_interact(0, 3.5e9, &m, &m, &m, &ui, &prop, &u32, &u32);
    quadriga_lib::ray_state_update(0, 3.5e9, &m, &m, &m, &m, &u32, &col, &m, &u8, &prop, &sc, &sc);
    T g = quadriga_lib::medium_gain(prop, 1, 1, 3.5e9);
    g = quadriga_lib::interface_gain(prop, 1, 3.5e9);
    g = quadriga_lib::refractive_index(prop, 1, 3.5e9);
    quadriga_lib::write_png("a.png", m, "jet", 0, 1);
    (void)g;
    (void)ad;
}

// The single and the vector forms of the antenna array functions share one name. Calls as users write them, with
// braced optional inputs and nullptr outputs (which need the template argument), select the intended form; compiled
// and linked only
static void calls_of_overloads()
{
    quadriga_lib::arrayant<double> a;
    std::vector<quadriga_lib::arrayant<double>> av;
    arma::vec bw;
    arma::mat bw_m, az, el, V_re, V_im, H_re, H_im;
    arma::cube Vf_re, Vf_im, Hf_re, Hf_im;

    bw = quadriga_lib::calc_directivity(a);
    bw_m = quadriga_lib::calc_directivity(av, {0, 1});
    quadriga_lib::calc_beamwidth(a, {}, 3.0, &bw);
    quadriga_lib::calc_beamwidth(av, {}, 3.0, &bw_m);
    quadriga_lib::calc_beamwidth<double>(a, {}, 3.0, nullptr, &bw);
    quadriga_lib::calc_beamwidth<double>(av, {}, 3.0, nullptr, &bw_m);
    a = quadriga_lib::apply_port_weights(a);
    av = quadriga_lib::apply_port_weights(av, {}, {}, {1.0e9}, 343.0);
    a = quadriga_lib::join_arrays(a, a);
    av = quadriga_lib::join_arrays(av, av);
    quadriga_lib::copy_element(a, 0, {1, 2});
    quadriga_lib::copy_element(av, 0, {1});
    quadriga_lib::set_element_pos(av, arma::mat(3, 2), {0, 1});
    quadriga_lib::rotate_pattern(a, {0.0, 0.1, 0.0});
    quadriga_lib::rotate_pattern(av, {0.0, 0.1, 0.0}, 1, {0});
    quadriga_lib::interpolate_pattern(a, az, el, V_re, V_im, H_re, H_im, {0});
    quadriga_lib::interpolate_pattern(av, az, el, Vf_re, Vf_im, Hf_re, Hf_im, {1.0e9}, {0});
    quadriga_lib::interpolate_pattern<double>(a, az, el, V_re, V_im, H_re, H_im, {}, {}, {}, nullptr, nullptr, nullptr,
                                              nullptr, nullptr, nullptr, 1);
    quadriga_lib::interpolate_pattern<double>(av, az, el, Vf_re, Vf_im, Hf_re, Hf_im, {}, {}, {}, {}, nullptr, nullptr, 1);
    std::string err = quadriga_lib::is_valid(av);
    unsigned id = quadriga_lib::qdant_write("a.qdant", a, 0, {});
    quadriga_lib::qdant_write("a.qdant", av);
    (void)err;
    (void)id;
}

TEST_CASE("Scalar argument types - literals of any type with float data")
{
    volatile bool never = false;
    if (never)
    {
        calls_with_literals();
        calls_of_overloads();
    }

    // The converted values arrive: int and double literals give the same result as float ones
    quadriga_lib::arrayant<T> a = quadriga_lib::generate_antenna<T>("custom", 10, NAN, {60, 30.0});
    quadriga_lib::arrayant<T> b = quadriga_lib::generate_antenna<T>("custom", 10.0f, NAN, {60.0f, 30.0f});
    CHECK(a.azimuth_grid.n_elem == 37);
    CHECK(arma::approx_equal(a.e_theta_re, b.e_theta_re, "absdiff", 0.0));

    arma::Col<T> bw_int, bw_float, bw_3dB;
    quadriga_lib::calc_beamwidth(a, arma::uvec(), 10, &bw_int);
    quadriga_lib::calc_beamwidth(a, arma::uvec(), 10.0f, &bw_float);
    CHECK(arma::approx_equal(bw_int, bw_float, "absdiff", 0.0));

    // The values are used: the 3 dB width is the requested 60 degree (within the interpolation error of the
    // 10 degree grid), and the 10 dB threshold gives a wider beam than the default of 3 dB
    quadriga_lib::calc_beamwidth(a, arma::uvec(), 3, &bw_3dB);
    CHECK(std::abs(bw_3dB(0) - T(60)) < T(1));
    CHECK(bw_int(0) > T(1.5) * bw_3dB(0));

    // dtype defaults to double for the generators
    quadriga_lib::arrayant<double> d = quadriga_lib::generate_antenna("omni", 10);
    CHECK(d.azimuth_grid.n_elem == 37);
}
