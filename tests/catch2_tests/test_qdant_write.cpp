// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// File output of qdant_write, checked by reading back with qdant_read: round trips of single- and multi-element
// arrays, IDs and layouts when appending and overwriting, accuracy of low-level and very large pattern data,
// single-precision accuracy of positions and coupling, locale-independent number text, namespace prefixes.
// Each case writes its own file in the working folder and removes it.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <clocale>
#include <cmath>
#include <filesystem>

TEST_CASE("Arrayant writing to QDANT file - Minimal test")
{
    std::remove("test.qdant");

    float pi = arma::datum::pi;

    quadriga_lib::arrayant<float> ant;
    ant.azimuth_grid = {-0.75f * pi, 0.0f, 0.75f * pi, pi};
    ant.elevation_grid = {-0.45f * pi, 0.0f, 0.45f * pi};

    arma::mat A = arma::linspace(1.0, 12.0, 12);
    A.reshape(3, 4);

    arma::fcube B;
    B.zeros(3, 4, 1);
    B.slice(0) = arma::conv_to<arma::fmat>::from(A);

    ant.e_theta_re = B * 0.5f;
    ant.e_theta_im = B * 0.002f;
    ant.e_phi_re = -B;
    ant.e_phi_im = -B * 0.001f;

    arma::fmat C = {1.0f, 2.0f, 4.0f};
    ant.element_pos = C.t();

    ant.port_weights_re = {1.0f};
    ant.port_weights_im = {0.1f};
    ant.center_freq = 2.0e9f;
    ant.name = "name";

    unsigned id = quadriga_lib::qdant_write("test.qdant", ant);
    CHECK(id == 1);

    auto antI = quadriga_lib::qdant_read<float>("test.qdant", {1})[0];

    CHECK(arma::approx_equal(antI.azimuth_grid, ant.azimuth_grid, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.elevation_grid, ant.elevation_grid, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.e_theta_re, ant.e_theta_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_theta_im, ant.e_theta_im, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_phi_re, ant.e_phi_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_phi_im, ant.e_phi_im, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.element_pos, ant.element_pos, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.port_weights_re, ant.port_weights_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.port_weights_im, ant.port_weights_im, "absdiff", 1e-4));
    CHECK(std::abs(antI.center_freq - ant.center_freq) < 0.1f);
    CHECK(antI.name == ant.name);

    std::remove("test.qdant");
}

TEST_CASE("Arrayant writing/reading from QDANT file - Multi-Element Array")
{
    double pi = arma::datum::pi;

    quadriga_lib::arrayant<double> ant;
    ant.name = "bla";
    ant.elevation_grid = arma::linspace<arma::vec>(-90.0, 90.0, 5) * pi / 180.0;
    ant.azimuth_grid = {-pi, 0.0f, 0.5f * pi};

    ant.e_theta_re = arma::cube(5, 3, 2, arma::fill::value(1.0));
    ant.e_theta_im = arma::cube(5, 3, 2, arma::fill::value(2.0));
    ant.e_phi_re = arma::cube(5, 3, 2, arma::fill::value(3.0));
    ant.e_phi_im = arma::cube(5, 3, 2, arma::fill::value(4.0));

    ant.e_theta_re.slice(1).fill(5.0);
    ant.e_theta_im.slice(1).fill(6.0);
    ant.e_phi_re.slice(1).fill(7.0);
    ant.e_phi_im.slice(1).fill(8.0);

    arma::mat A = arma::reshape(arma::linspace<arma::vec>(1.0, 6.0, 6), 2, 3);
    arma::mat B = arma::reshape(arma::linspace<arma::vec>(2.0, 7.0, 6), 2, 3);
    ant.port_weights_re = A;
    ant.port_weights_im = B;

    unsigned id = quadriga_lib::qdant_write("test2.qdant", ant);
    CHECK(id == 1);

    auto x = quadriga_lib::qdant_read<double>("test2.qdant", {1})[0];
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(1.0)), x.e_theta_re.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(2.0)), x.e_theta_im.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(3.0)), x.e_phi_re.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(4.0)), x.e_phi_im.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(5.0)), x.e_theta_re.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(6.0)), x.e_theta_im.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(7.0)), x.e_phi_re.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(8.0)), x.e_phi_im.slice(1), "absdiff", 2e-5));
    CHECK(arma::approx_equal(A, x.port_weights_re, "absdiff", 1e-5));
    CHECK(arma::approx_equal(B, x.port_weights_im, "absdiff", 1e-5));

    std::remove("test2.qdant");
}

TEST_CASE("Arrayant writing to QDANT file - Complex test")
{
    std::ofstream f;
    f.open("test.qdant");
    f << "<qdant><arrayant>" << std::endl;
    f << "<name>bla</name>" << std::endl;
    f << "<ElevationGrid>-90 -45 0 45 90</ElevationGrid>" << std::endl;
    f << "<AzimuthGrid>-180 0 90</AzimuthGrid>" << std::endl;
    f << "<EthetaMag>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15</EthetaMag>" << std::endl;
    f << "</arrayant></qdant>" << std::endl;
    f.close();

    double pi = arma::datum::pi;

    quadriga_lib::arrayant<double> ant;
    ant.azimuth_grid = {-0.75 * pi, 0.0, 0.75 * pi, pi};
    ant.elevation_grid = {-0.45 * pi, 0.0, 0.45 * pi};

    arma::mat A = arma::linspace(1.0, 12.0, 12);
    A.reshape(3, 4);

    arma::cube B;
    B.zeros(3, 4, 1);
    B.slice(0) = A;

    ant.e_theta_re = B * 0.5;
    ant.e_theta_im = B * 0.002;
    ant.e_phi_re = -B;
    ant.e_phi_im = -B * 0.001;

    arma::mat C = {1.0, 2.0, 4.0};
    ant.element_pos = C.t();

    ant.port_weights_re = {1.0};
    ant.port_weights_im = {0.1};
    ant.center_freq = 2.0e9;
    ant.name = "name";

    // Append new antenna
    unsigned id = quadriga_lib::qdant_write("test.qdant", ant);

    CHECK(id == 2);

    arma::Mat<unsigned> layout, layout2;

    auto antI = quadriga_lib::qdant_read<float>("test.qdant", {1}, &layout2)[0];

    CHECK(antI.name == "bla");

    antI = quadriga_lib::qdant_read<float>("test.qdant", {2})[0];
    CHECK(antI.name == "name");

    layout = {1, 2};
    CHECK(arma::all(arma::vectorise(layout2 - layout) == 0));

    id = quadriga_lib::qdant_write("test.qdant", ant, 112);
    CHECK(id == 112);

    // Layout can only contain valid entries (error expected)
    layout = {{1, 2, 112}, {112, 112, 6}};
    REQUIRE_THROWS_AS(quadriga_lib::qdant_write("test.qdant", ant, 5, layout), std::invalid_argument);

    // A file that cannot be written (the folder does not exist)
    REQUIRE_THROWS_AS(quadriga_lib::qdant_write("test_no_such_dir_qdant/test.qdant", ant), std::runtime_error);

    id = quadriga_lib::qdant_write("test.qdant", ant, 6, layout);
    CHECK(id == 6);

    antI = quadriga_lib::qdant_read<float>("test.qdant", {6}, &layout2)[0];

    CHECK(arma::all(arma::vectorise(layout2 - layout) == 0));

    std::remove("test.qdant");
}

// Largest absolute difference relative to the largest absolute value of the reference
template <typename T>
static double qdant_rel_err(const arma::Cube<T> &ref, const arma::Cube<T> &x)
{
    if (arma::size(ref) != arma::size(x))
        return INFINITY;
    arma::Col<double> r = arma::conv_to<arma::Col<double>>::from(arma::vectorise(ref));
    arma::Col<double> d = r - arma::conv_to<arma::Col<double>>::from(arma::vectorise(x));
    return arma::abs(d).max() / std::max(arma::abs(r).max(), 1e-300);
}

template <typename T>
static double qdant_rel_err(const arma::Mat<T> &ref, const arma::Mat<T> &x)
{
    return qdant_rel_err(arma::Cube<T>(ref.memptr(), ref.n_rows, ref.n_cols, 1), arma::Cube<T>(x.memptr(), x.n_rows, x.n_cols, 1));
}

// Largest error of the complex field (both components) relative to the peak of the element, over all elements
template <typename T>
static double qdant_err_element(const quadriga_lib::arrayant<T> &ref, const quadriga_lib::arrayant<T> &x)
{
    if (arma::size(ref.e_theta_re) != arma::size(x.e_theta_re))
        return INFINITY;
    double err = 0.0;
    for (arma::uword e = 0; e < ref.n_elements(); ++e)
    {
        arma::cx_mat rt(arma::conv_to<arma::mat>::from(ref.e_theta_re.slice(e)), arma::conv_to<arma::mat>::from(ref.e_theta_im.slice(e)));
        arma::cx_mat rp(arma::conv_to<arma::mat>::from(ref.e_phi_re.slice(e)), arma::conv_to<arma::mat>::from(ref.e_phi_im.slice(e)));
        arma::cx_mat xt(arma::conv_to<arma::mat>::from(x.e_theta_re.slice(e)), arma::conv_to<arma::mat>::from(x.e_theta_im.slice(e)));
        arma::cx_mat xp(arma::conv_to<arma::mat>::from(x.e_phi_re.slice(e)), arma::conv_to<arma::mat>::from(x.e_phi_im.slice(e)));
        double peak = std::max(arma::abs(rt).max(), arma::abs(rp).max());
        if (peak == 0.0)
            continue;
        arma::mat dt = arma::abs(xt - rt), dp = arma::abs(xp - rp);
        for (arma::uword i = 0; i < dt.n_elem; ++i)
            err = std::max(err, std::max(dt(i), dp(i)) / peak);
    }
    return err;
}

// The file is accurate to 6e-8 of the element peak; an element at a low level is quantized relative to its own peak
TEST_CASE("Arrayant writing to QDANT file - Low-level pattern data is kept")
{
    const std::string fn = "test_qdant_low_level.qdant";

    for (double scale : {1e-6, 1e-12, 1e-30})
    {
        std::remove(fn.c_str());
        auto a = quadriga_lib::generate_antenna<double>("custom", 10.0, NAN, {90.0, 90.0, 10.0});
        a.e_theta_re *= scale;
        quadriga_lib::qdant_write(fn, a);
        auto b = quadriga_lib::qdant_read<double>(fn, {1})[0];
        CHECK(qdant_err_element(a, b) < 1.2e-7);
        CHECK(arma::abs(arma::vectorise(b.e_theta_re)).max() > 0.5 * scale);
    }

    // One sample at about -110 dB (about -120 dB below the peak) and one exact zero in a normal pattern
    {
        std::remove(fn.c_str());
        auto a = quadriga_lib::generate_antenna<double>("custom", 10.0, NAN, {90.0, 90.0, 10.0});
        a.e_theta_re(1, 0, 0) = 3e-6;
        a.e_theta_re(2, 0, 0) = 0.0;
        quadriga_lib::qdant_write(fn, a);
        auto b = quadriga_lib::qdant_read<double>(fn, {1})[0];
        CHECK(qdant_err_element(a, b) < 1.2e-7);
        CHECK(std::abs(b.e_theta_re(1, 0, 0) / 3e-6 - 1.0) < 0.06); // kept, with at most 0.5 dB rounding
        CHECK(b.e_theta_re(2, 0, 0) == 0.0);
        CHECK(b.e_theta_im(2, 0, 0) == 0.0);
    }

    // Element 2 of 3 is zero everywhere, element 3 is at -200 dB (accurate relative to its own peak)
    {
        std::remove(fn.c_str());
        auto a = quadriga_lib::generate_antenna<double>("custom", 10.0, NAN, {90.0, 90.0, 10.0});
        quadriga_lib::copy_element(a, 0, arma::uvec{1, 2});
        a.e_theta_re.slice(1).zeros(), a.e_theta_im.slice(1).zeros(), a.e_phi_re.slice(1).zeros(), a.e_phi_im.slice(1).zeros();
        a.e_theta_re.slice(2) *= 1e-10;
        quadriga_lib::qdant_write(fn, a);
        auto b = quadriga_lib::qdant_read<double>(fn, {1})[0];
        REQUIRE(b.n_elements() == 3);
        CHECK(arma::all(arma::vectorise(b.e_theta_re.slice(1)) == 0.0));
        CHECK(qdant_err_element(a, b) < 1.2e-7);
        CHECK(arma::abs(arma::vectorise(b.e_theta_re.slice(2))).max() > 0.5e-10);
    }

    // Float pattern at 1e-20 (power 1e-40 is below the smallest normal float); the float reader parses the dB values
    // (about -400) as float, whose resolution there is 3e-5 dB, i.e. 2e-6 of the amplitude
    {
        std::remove(fn.c_str());
        auto a = quadriga_lib::generate_antenna<float>("custom", 10.0f, NAN, {90.0f, 90.0f, 10.0});
        a.e_theta_re *= 1e-20f;
        quadriga_lib::qdant_write(fn, a);
        auto b = quadriga_lib::qdant_read<float>(fn, {1})[0];
        CHECK(qdant_err_element(a, b) < 1e-5);
    }

    std::remove(fn.c_str());
}

// The file has single-precision-equivalent accuracy: positions and coupling (magnitude, phase) as floats, pattern
// samples within 6e-8 of the element peak
TEST_CASE("Arrayant writing to QDANT file - Single-precision accuracy of positions, coupling and pattern")
{
    const std::string fn = "test_qdant_precision.qdant";
    arma::arma_rng::set_seed(38);

    // Double: 3 elements, 2 ports, random polarimetric pattern (peak about +10 dB)
    {
        std::remove(fn.c_str());
        quadriga_lib::arrayant<double> a;
        a.name = "precision";
        a.azimuth_grid = arma::linspace<arma::vec>(-arma::datum::pi, arma::datum::pi * 0.9, 20);
        a.elevation_grid = arma::linspace<arma::vec>(-arma::datum::pi / 2.0, arma::datum::pi / 2.0, 7);
        a.e_theta_re.randn(7, 20, 3), a.e_theta_im.randn(7, 20, 3), a.e_phi_re.randn(7, 20, 3), a.e_phi_im.randn(7, 20, 3);
        a.e_phi_re.slice(2) *= 1e-6, a.e_phi_im.slice(2) *= 1e-6, a.e_theta_re.slice(2) *= 1e-6, a.e_theta_im.slice(2) *= 1e-6;
        a.element_pos.randn(3, 3);
        a.element_pos(0, 1) = 123.456789, a.element_pos(1, 1) = 0.000123456789;
        a.port_weights_re.randn(3, 2), a.port_weights_im.randn(3, 2);
        a.port_weights_re(0, 0) = 0.123456789;
        a.center_freq = 3.456789012e9;
        quadriga_lib::qdant_write(fn, a);

        auto b = quadriga_lib::qdant_read<double>(fn, {1})[0];

        // Positions exact as float: the value read back rounds to the same float
        arma::fmat pos_float = arma::conv_to<arma::fmat>::from(a.element_pos);
        CHECK(arma::approx_equal(arma::conv_to<arma::fmat>::from(b.element_pos), pos_float, "absdiff", 0.0f));
        CHECK((float)b.element_pos(0, 1) == 123.456789f);
        CHECK(std::abs(b.element_pos(1, 1) / 0.000123456789 - 1.0) < 1e-7);

        // Coupling from float magnitude and phase: each entry within 3e-7 of its magnitude
        arma::cx_mat ca(a.port_weights_re, a.port_weights_im), cb(b.port_weights_re, b.port_weights_im);
        CHECK(arma::max(arma::vectorise(arma::abs(cb - ca) / arma::abs(ca))) < 3e-7);

        // Pattern within 1.2e-7 of the element peak (float epsilon), also for the element at -120 dB; samples far
        // below the peak are not accurate to 1.2e-7 of their own amplitude
        CHECK(qdant_err_element(a, b) < 1.2e-7);
        CHECK(b.center_freq == a.center_freq);
    }

    // Float: positions exact; the pattern error includes the float arithmetic of the reader
    {
        std::remove(fn.c_str());
        quadriga_lib::arrayant<float> a;
        a.azimuth_grid = arma::linspace<arma::fvec>(-arma::fdatum::pi, arma::fdatum::pi * 0.9f, 20);
        a.elevation_grid = arma::linspace<arma::fvec>(-arma::fdatum::pi / 2.0f, arma::fdatum::pi / 2.0f, 7);
        a.e_theta_re.randn(7, 20, 2), a.e_theta_im.randn(7, 20, 2), a.e_phi_re.randn(7, 20, 2), a.e_phi_im.randn(7, 20, 2);
        a.element_pos.randn(3, 2);
        a.element_pos(0, 1) = 123.456789f;
        a.port_weights_re.randn(2, 2), a.port_weights_im.randn(2, 2);
        quadriga_lib::qdant_write(fn, a);

        auto b = quadriga_lib::qdant_read<float>(fn, {1})[0];
        CHECK(arma::approx_equal(b.element_pos, a.element_pos, "absdiff", 0.0f));
        CHECK(qdant_rel_err(a.port_weights_re, b.port_weights_re) < 1e-6);
        CHECK(qdant_rel_err(a.port_weights_im, b.port_weights_im) < 1e-6);
        CHECK(qdant_err_element(a, b) < 1e-6);
    }

    std::remove(fn.c_str());
}

// Number text of the writer: the shortest float text in the form std::to_chars writes (fixed or scientific, whichever
// is shorter), '.' as decimal point also when the C locale uses a comma
TEST_CASE("Arrayant writing to QDANT file - Number text and locale")
{
    const std::string fn = "test_qdant_numtext.qdant", fn_loc = "test_qdant_numtext_locale.qdant";
    auto read_text = [](const std::string &f)
    {
        std::ifstream in(f, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };

    auto a = quadriga_lib::generate_antenna<double>("omni", 10.0);
    quadriga_lib::copy_element(a, 0, {1, 2});
    a.element_pos = {{1200.0, 0.0001, 2.5e-7}, {-0.0, 1e-5, 1e15}, {123456.0, 0.1, 16777216.0}};
    std::remove(fn.c_str());
    quadriga_lib::qdant_write(fn, a);
    const std::string txt = read_text(fn);
    CHECK(txt.find("<ElementPosition>1200,-0,123456 1e-04,1e-05,0.1 2.5e-07,1e+15,16777216</ElementPosition>") != std::string::npos);

    // A locale with a decimal comma (if one is installed) must not change a single byte
    const std::string old_locale = std::setlocale(LC_NUMERIC, nullptr);
    const char *comma_locale = nullptr;
    for (const char *l : {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "fr_FR.UTF-8", "fr_FR.utf8", "ru_RU.UTF-8"})
        if (std::setlocale(LC_NUMERIC, l) != nullptr && std::string(std::localeconv()->decimal_point) == ",")
        {
            comma_locale = l;
            break;
        }
    if (comma_locale != nullptr)
    {
        std::remove(fn_loc.c_str());
        quadriga_lib::qdant_write(fn_loc, a);
        std::setlocale(LC_NUMERIC, old_locale.c_str());
        INFO("locale " << comma_locale);
        CHECK(read_text(fn_loc) == txt);
        std::remove(fn_loc.c_str());
    }
    else
    {
        std::setlocale(LC_NUMERIC, old_locale.c_str());
        std::remove(fn.c_str()); // SKIP leaves the test case at once
        SKIP("No locale with a decimal comma installed; the locale part was skipped.");
    }
    std::remove(fn.c_str());
}

// Documented behaviour: a write without a layout replaces the layout in the file by all IDs in file order
TEST_CASE("Arrayant writing to QDANT file - A write without layout resets the layout")
{
    const std::string fn = "test_qdant_layout_reset.qdant";
    std::remove(fn.c_str());

    auto a = quadriga_lib::generate_antenna<double>("omni", 30.0);
    quadriga_lib::qdant_write(fn, a, 1), quadriga_lib::qdant_write(fn, a, 2), quadriga_lib::qdant_write(fn, a, 3);
    arma::u32_mat L = {{1, 2}, {3, 3}};
    quadriga_lib::qdant_write(fn, a, 3, L);

    arma::u32_mat L1;
    quadriga_lib::qdant_read<double>(fn, {1}, &L1);
    REQUIRE(arma::size(L1) == arma::size(L));
    CHECK(arma::all(arma::vectorise(L1 == L)));

    // Append without layout
    CHECK(quadriga_lib::qdant_write(fn, a) == 4);
    quadriga_lib::qdant_read<double>(fn, {1}, &L1);
    arma::u32_mat L_reset = {{1, 2, 3, 4}};
    REQUIRE(arma::size(L1) == arma::size(L_reset));
    CHECK(arma::all(arma::vectorise(L1 == L_reset)));

    // Overwriting an entry without layout also resets a custom layout
    quadriga_lib::qdant_write(fn, a, 4, L);
    quadriga_lib::qdant_write(fn, a, 2);
    quadriga_lib::qdant_read<double>(fn, {1}, &L1);
    REQUIRE(arma::size(L1) == arma::size(L_reset));
    CHECK(arma::all(arma::vectorise(L1 == L_reset)));

    std::remove(fn.c_str());
}

TEST_CASE("Arrayant writing to QDANT file - append to a file with a namespace prefix")
{
    // A file whose root declares a prefix (xmlns:xx) has prefixed node names; an appended or overwritten entry must
    // use the same prefix, or it cannot be read back
    const std::string fn = "test_qdant_prefix.qdant";
    std::remove(fn.c_str());
    {
        std::ofstream f(fn);
        f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?><qdant xmlns:xx=\"http://www.quadriga-channel-model.de\">\n"
          << "<xx:arrayant id=\"1\">\n<xx:name>first</xx:name>\n"
          << "<xx:ElevationGrid>-90 0 90</xx:ElevationGrid>\n<xx:AzimuthGrid>-180 0 90</xx:AzimuthGrid>\n"
          << "<xx:EthetaMag>0 0 0 0 0 0 0 0 0</xx:EthetaMag>\n</xx:arrayant></qdant>\n";
    }

    auto ant = quadriga_lib::generate_antenna<double>("xpol", 10.0);
    ant.name = "second";
    unsigned id = quadriga_lib::qdant_write(fn, ant); // appended with ID 2
    CHECK(id == 2);

    auto first = quadriga_lib::qdant_read<double>(fn, {1})[0];
    CHECK(first.name == "first");
    auto second = quadriga_lib::qdant_read<double>(fn, {2})[0];
    CHECK(second.name == "second");
    CHECK(second.e_theta_re.n_slices == 2);
    CHECK(arma::approx_equal(second.e_theta_re, ant.e_theta_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(second.e_phi_re, ant.e_phi_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(second.azimuth_grid, ant.azimuth_grid, "absdiff", 1e-6));

    // Overwrite entry 1 in place
    ant.name = "replaced";
    CHECK(quadriga_lib::qdant_write(fn, ant, 1) == 1);
    CHECK(quadriga_lib::qdant_read<double>(fn, {1})[0].name == "replaced");
    CHECK(quadriga_lib::qdant_read<double>(fn, {2})[0].name == "second");

    // Every node in the file carries the prefix
    std::ifstream f(fn);
    std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(txt.find("<name>") == std::string::npos);
    CHECK(txt.find("<arrayant") == std::string::npos);
    CHECK(txt.find("<EthetaMag") == std::string::npos);
    std::remove(fn.c_str());
}

TEST_CASE("Arrayant writing to QDANT file - Field values with power above the double range")
{
    // |E|^2 overflows double for amplitudes above about 1e154; the written dB values must stay finite
    const std::string fn = "test_qdant_large.qdant";
    std::remove(fn.c_str());
    auto a = quadriga_lib::generate_antenna<double>("custom", 30.0, NAN, {90.0, 90.0});
    a.e_theta_re *= 1.0e160;
    a.e_phi_im = a.e_theta_re * 0.5;
    quadriga_lib::qdant_write(fn, a, 1);
    auto b = quadriga_lib::qdant_read<double>(fn, {1})[0];
    std::remove(fn.c_str());
    REQUIRE(b.e_theta_re.is_finite());
    CHECK(arma::approx_equal(b.e_theta_re * 1.0e-160, a.e_theta_re * 1.0e-160, "absdiff", 1.0e-6));
    CHECK(arma::approx_equal(b.e_phi_im * 1.0e-160, a.e_phi_im * 1.0e-160, "absdiff", 1.0e-6));
}

// ---------------------------------------------------------------------------------------------------------------------
// Tests of quadriga_lib::qdant_write and qdant_read of all entries: round trips of multi-frequency speaker
// models and other arrayants, the layout output, overwriting, agreement with qdant_read per entry, and errors.
// QDANT is XML text, so values round-trip only to the printed precision (tolerances tol_d and tol_f).

// QDANT is XML text, so roundtrip precision is limited by text formatting.
// Use a tolerance that accounts for this.
static const double tol_d = 1.0e-5;
static const float tol_f = 1.0e-4f;

TEST_CASE("QDANT vector - Write/Read roundtrip with speaker model")
{
    std::string fn = "test_speaker_roundtrip.qdant";

    // Generate a small speaker model: 5 frequencies, coarse grid
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "hemisphere",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 5);

    REQUIRE_NOTHROW(quadriga_lib::qdant_write(fn, spk));
    CHECK(std::filesystem::exists(fn));

    arma::u32_mat layout;
    auto spk_read = quadriga_lib::qdant_read<double>(fn, {}, &layout);

    REQUIRE(spk_read.size() == spk.size());

    // Layout should be [n_freq, 1] with entries 1...n_freq
    REQUIRE(layout.n_rows == 5);
    REQUIRE(layout.n_cols == 1);
    for (arma::uword i = 0; i < 5; ++i)
        CHECK(layout(i, 0) == (unsigned)(i + 1));

    for (size_t i = 0; i < spk.size(); ++i)
    {
        INFO("Frequency index: " << i);

        // Grid sizes must match exactly
        CHECK(spk_read[i].n_azimuth() == spk[i].n_azimuth());
        CHECK(spk_read[i].n_elevation() == spk[i].n_elevation());
        CHECK(spk_read[i].n_elements() == spk[i].n_elements());
        CHECK(spk_read[i].n_ports() == spk[i].n_ports());

        CHECK(std::abs(spk_read[i].center_freq - spk[i].center_freq) < tol_d);

        CHECK(spk_read[i].name == spk[i].name);

        CHECK(arma::approx_equal(spk_read[i].e_theta_re, spk[i].e_theta_re, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].e_theta_im, spk[i].e_theta_im, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].e_phi_re, spk[i].e_phi_re, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].e_phi_im, spk[i].e_phi_im, "absdiff", tol_d));

        CHECK(arma::approx_equal(spk_read[i].azimuth_grid, spk[i].azimuth_grid, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].elevation_grid, spk[i].elevation_grid, "absdiff", tol_d));

        CHECK(arma::approx_equal(spk_read[i].element_pos, spk[i].element_pos, "absdiff", tol_d));

        CHECK(arma::approx_equal(spk_read[i].port_weights_re, spk[i].port_weights_re, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].port_weights_im, spk[i].port_weights_im, "absdiff", tol_d));
    }

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - Single entry roundtrip")
{
    std::string fn = "test_speaker_single.qdant";

    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "monopole",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    quadriga_lib::qdant_write(fn, spk);

    arma::u32_mat layout;
    auto spk_read = quadriga_lib::qdant_read<double>(fn, {}, &layout);

    REQUIRE(spk_read.size() == 1);
    REQUIRE(layout.n_rows == 1);
    REQUIRE(layout.n_cols == 1);
    CHECK(layout(0, 0) == 1);

    CHECK(std::abs(spk_read[0].center_freq - 1000.0) < tol_d);
    CHECK(arma::approx_equal(spk_read[0].e_theta_re, spk[0].e_theta_re, "absdiff", tol_d));

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - Float template roundtrip")
{
    std::string fn = "test_speaker_float.qdant";

    arma::fvec freqs = {200.0f, 2000.0f};
    auto spk = quadriga_lib::generate_speaker<float>("horn", 0.05f, 80.0f, 12000.0f,
                                                      12.0f, 12.0f, 85.0f, "monopole",
                                                      0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);
    REQUIRE(spk.size() == 2);

    quadriga_lib::qdant_write(fn, spk);
    auto spk_read = quadriga_lib::qdant_read<float>(fn);

    REQUIRE(spk_read.size() == 2);

    for (size_t i = 0; i < spk.size(); ++i)
    {
        CHECK(std::abs(spk_read[i].center_freq - spk[i].center_freq) < tol_f);
        CHECK(arma::approx_equal(spk_read[i].e_theta_re, spk[i].e_theta_re, "absdiff", tol_f));
    }

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - an existing file is not overwritten")
{
    std::string fn = "test_speaker_overwrite.qdant";

    // Write a 3-entry file
    arma::vec freqs3 = {100.0, 500.0, 1000.0};
    auto spk3 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                        12.0, 12.0, 85.0, "monopole",
                                                        0.0, 0.0, 0.0, 0.15, 0.25, freqs3, 10.0);
    quadriga_lib::qdant_write(fn, spk3);

    auto read3 = quadriga_lib::qdant_read<double>(fn);
    REQUIRE(read3.size() == 3);

    // Writing a 2-entry vector to the existing file throws and leaves the file unchanged
    arma::vec freqs2 = {2000.0, 8000.0};
    auto spk2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                        12.0, 12.0, 85.0, "monopole",
                                                        0.0, 0.0, 0.0, 0.15, 0.25, freqs2, 10.0);
    CHECK_THROWS_AS(quadriga_lib::qdant_write(fn, spk2), std::runtime_error);

    auto read2 = quadriga_lib::qdant_read<double>(fn);
    REQUIRE(read2.size() == 3);
    CHECK(std::abs(read2[0].center_freq - 100.0) < tol_d);
    CHECK(std::abs(read2[2].center_freq - 1000.0) < tol_d);

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - All driver/radiation types survive roundtrip")
{
    std::string fn = "test_speaker_all_types.qdant";

    std::vector<std::string> drivers = {"piston", "horn", "omni"};
    std::vector<std::string> radiations = {"monopole", "hemisphere", "dipole", "cardioid"};
    arma::vec freqs = {500.0, 2000.0};

    for (const auto &drv : drivers)
    {
        for (const auto &rad : radiations)
        {
            INFO("driver=" << drv << " radiation=" << rad);

            auto spk = quadriga_lib::generate_speaker<double>(drv, 0.05, 80.0, 12000.0,
                                                               12.0, 12.0, 85.0, rad,
                                                               0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

            quadriga_lib::qdant_write(fn, spk);
            auto spk_read = quadriga_lib::qdant_read<double>(fn);

            REQUIRE(spk_read.size() == spk.size());

            for (size_t i = 0; i < spk.size(); ++i)
            {
                CHECK(arma::approx_equal(spk_read[i].e_theta_re, spk[i].e_theta_re, "absdiff", tol_d));
                CHECK(arma::approx_equal(spk_read[i].e_theta_im, spk[i].e_theta_im, "absdiff", tol_d));
            }

            std::filesystem::remove(fn);
        }
    }
}

TEST_CASE("QDANT vector - Roundtrip preserves frequency ordering")
{
    std::string fn = "test_speaker_ordering.qdant";

    // Non-uniform frequency spacing
    arma::vec freqs = {63.0, 250.0, 1000.0, 4000.0, 16000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 30.0, 20000.0,
                                                       12.0, 12.0, 85.0, "hemisphere",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::qdant_write(fn, spk);
    auto spk_read = quadriga_lib::qdant_read<double>(fn);

    REQUIRE(spk_read.size() == 5);

    // Frequencies should come back in the same order
    for (size_t i = 0; i < 5; ++i)
        CHECK(std::abs(spk_read[i].center_freq - freqs[i]) < tol_d);

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - Read of single IDs matches the read of all entries")
{
    std::string fn = "test_speaker_compat.qdant";

    arma::vec freqs = {100.0, 1000.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "hemisphere",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::qdant_write(fn, spk);

    // Read individual entries by ID
    for (size_t i = 0; i < spk.size(); ++i)
    {
        unsigned id = (unsigned)(i + 1);
        auto ant = quadriga_lib::qdant_read<double>(fn, {id})[0];

        CHECK(std::abs(ant.center_freq - spk[i].center_freq) < tol_d);
        CHECK(arma::approx_equal(ant.e_theta_re, spk[i].e_theta_re, "absdiff", tol_d));
        CHECK(ant.name == spk[i].name);
    }

    // Compare: the read of all entries should match the individual reads
    auto spk_multi = quadriga_lib::qdant_read<double>(fn);
    REQUIRE(spk_multi.size() == 3);

    for (size_t i = 0; i < 3; ++i)
    {
        auto ant_single = quadriga_lib::qdant_read<double>(fn, {i + 1})[0];
        CHECK(arma::approx_equal(spk_multi[i].e_theta_re, ant_single.e_theta_re, "absdiff", 1e-12));
        CHECK(std::abs(spk_multi[i].center_freq - ant_single.center_freq) < 1e-12);
    }

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - Write with auto-generated frequencies")
{
    std::string fn = "test_speaker_auto_freq.qdant";

    // Use default (auto third-octave) frequency generation
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 200.0, 8000.0,
                                                       12.0, 12.0, 85.0, "hemisphere");
    CHECK(spk.size() > 5); // Should have many third-octave bands

    quadriga_lib::qdant_write(fn, spk);
    auto spk_read = quadriga_lib::qdant_read<double>(fn);

    REQUIRE(spk_read.size() == spk.size());

    // Spot-check first and last entries
    CHECK(std::abs(spk_read.front().center_freq - spk.front().center_freq) < tol_d);
    CHECK(std::abs(spk_read.back().center_freq - spk.back().center_freq) < tol_d);
    CHECK(arma::approx_equal(spk_read.front().e_theta_re, spk.front().e_theta_re, "absdiff", tol_d));
    CHECK(arma::approx_equal(spk_read.back().e_theta_re, spk.back().e_theta_re, "absdiff", tol_d));

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - Write empty vector throws")
{
    std::string fn = "test_speaker_empty.qdant";
    std::vector<quadriga_lib::arrayant<double>> empty_vec;

    REQUIRE_THROWS_AS(quadriga_lib::qdant_write(fn, empty_vec), std::invalid_argument);

    CHECK(!std::filesystem::exists(fn));
}

TEST_CASE("QDANT vector - Write empty filename throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "monopole",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    REQUIRE_THROWS_AS(quadriga_lib::qdant_write("", spk), std::runtime_error);
}

TEST_CASE("QDANT vector - Read empty filename throws")
{
    REQUIRE_THROWS_AS(quadriga_lib::qdant_read<double>(""), std::runtime_error);
}

TEST_CASE("QDANT vector - Read non-existent file throws")
{
    REQUIRE_THROWS_AS(quadriga_lib::qdant_read<double>("does_not_exist_12345.qdant"), std::runtime_error);
}

TEST_CASE("QDANT vector - Read without layout pointer")
{
    std::string fn = "test_speaker_no_layout.qdant";

    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "monopole",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::qdant_write(fn, spk);

    // Read without requesting layout (nullptr default)
    auto spk_read = quadriga_lib::qdant_read<double>(fn);

    REQUIRE(spk_read.size() == 2);
    CHECK(std::abs(spk_read[0].center_freq - 500.0) < tol_d);
    CHECK(std::abs(spk_read[1].center_freq - 2000.0) < tol_d);

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - Non-speaker arrayant roundtrip")
{
    // Verify that write/read_multi works with any arrayant, not just speaker models
    std::string fn = "test_generic_multi.qdant";

    std::vector<quadriga_lib::arrayant<double>> vec;

    auto omni = quadriga_lib::generate_antenna<double>("omni", 10.0);
    omni.center_freq = 1000.0;
    vec.push_back(omni);

    auto dipole = quadriga_lib::generate_antenna<double>("dipole", 10.0);
    dipole.center_freq = 2000.0;
    vec.push_back(dipole);

    auto hw = quadriga_lib::generate_antenna<double>("half-wave-dipole", 10.0);
    hw.center_freq = 3000.0;
    vec.push_back(hw);

    quadriga_lib::qdant_write(fn, vec);
    auto vec_read = quadriga_lib::qdant_read<double>(fn);

    REQUIRE(vec_read.size() == 3);

    CHECK(vec_read[0].name == "omni");
    CHECK(vec_read[1].name == "dipole");
    CHECK(vec_read[2].name == "half-wave-dipole");

    CHECK(std::abs(vec_read[0].center_freq - 1000.0) < tol_d);
    CHECK(std::abs(vec_read[1].center_freq - 2000.0) < tol_d);
    CHECK(std::abs(vec_read[2].center_freq - 3000.0) < tol_d);

    CHECK(arma::approx_equal(vec_read[0].e_theta_re, omni.e_theta_re, "absdiff", tol_d));
    CHECK(arma::approx_equal(vec_read[1].e_theta_re, dipole.e_theta_re, "absdiff", tol_d));
    CHECK(arma::approx_equal(vec_read[2].e_theta_re, hw.e_theta_re, "absdiff", tol_d));

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - Large number of entries")
{
    std::string fn = "test_speaker_many.qdant";

    // Generate many frequency samples via auto third-octave over a wide range
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 30.0, 16000.0,
                                                       12.0, 12.0, 85.0, "monopole",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, {}, 10.0);

    CHECK(spk.size() >= 20); // Expect ~28 third-octave bands

    quadriga_lib::qdant_write(fn, spk);

    arma::u32_mat layout;
    auto spk_read = quadriga_lib::qdant_read<double>(fn, {}, &layout);

    REQUIRE(spk_read.size() == spk.size());
    REQUIRE(layout.n_rows == (arma::uword)spk.size());
    REQUIRE(layout.n_cols == 1);

    for (arma::uword i = 0; i < layout.n_rows; ++i)
        CHECK(layout(i, 0) == (unsigned)(i + 1));

    // Spot-check middle entry
    size_t mid = spk.size() / 2;
    CHECK(std::abs(spk_read[mid].center_freq - spk[mid].center_freq) < tol_d);
    CHECK(arma::approx_equal(spk_read[mid].e_theta_re, spk[mid].e_theta_re, "absdiff", tol_d));

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT vector - Invalid entry keeps existing file")
{
    // Validation must happen before an existing file is removed
    std::string fn = "test_qdant_multi_invalid_keep.qdant";
    auto a = quadriga_lib::generate_antenna<double>("omni", 10.0);
    quadriga_lib::qdant_write(fn, a);
    REQUIRE(std::filesystem::exists(fn));
    auto size_before = std::filesystem::file_size(fn);

    std::vector<quadriga_lib::arrayant<double>> vec(2, a);
    vec[1].e_theta_im.reset(); // invalid entry
    REQUIRE_THROWS_AS(quadriga_lib::qdant_write(fn, vec), std::invalid_argument);

    REQUIRE(std::filesystem::exists(fn));
    CHECK(std::filesystem::file_size(fn) == size_before);
    auto b = quadriga_lib::qdant_read<double>(fn, {1})[0];
    CHECK(arma::approx_equal(b.e_theta_re, a.e_theta_re, "absdiff", 1e-6));

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT write - Port weights use the format's coupling tags")
{
    const std::string fn = "test_qdant_coupling_tags.qdant";
    std::remove(fn.c_str());
    auto ant = quadriga_lib::generate_antenna<double>("xpol", 30.0);
    ant.port_weights_re = arma::mat({{0.5, 0.5}, {0.5, -0.5}});
    ant.port_weights_im = arma::mat(2, 2, arma::fill::zeros);
    quadriga_lib::qdant_write(fn, ant);

    std::ifstream in(fn);
    std::stringstream text;
    text << in.rdbuf();
    CHECK(text.str().find("CouplingAbs>") != std::string::npos);
    CHECK(text.str().find("CouplingPhase>") != std::string::npos);
    CHECK(text.str().find("port_weights") == std::string::npos);
    std::remove(fn.c_str());
}
