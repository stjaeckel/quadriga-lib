// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <clocale>

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

    ant.coupling_re = {1.0f};
    ant.coupling_im = {0.1f};
    ant.center_frequency = 2.0e9f;
    ant.name = "name";

    unsigned id = ant.qdant_write("test.qdant");
    CHECK(id == 1);

    // Load file again and compare
    auto antI = quadriga_lib::qdant_read<float>("test.qdant");

    CHECK(arma::approx_equal(antI.azimuth_grid, ant.azimuth_grid, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.elevation_grid, ant.elevation_grid, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.e_theta_re, ant.e_theta_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_theta_im, ant.e_theta_im, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_phi_re, ant.e_phi_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_phi_im, ant.e_phi_im, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.element_pos, ant.element_pos, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.coupling_re, ant.coupling_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.coupling_im, ant.coupling_im, "absdiff", 1e-4));
    CHECK(std::abs(antI.center_frequency - ant.center_frequency) < 0.1f);
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
    ant.coupling_re = A;
    ant.coupling_im = B;

    unsigned id = ant.qdant_write("test2.qdant");
    CHECK(id == 1);

    auto x = quadriga_lib::qdant_read<double>("test2.qdant");
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(1.0)), x.e_theta_re.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(2.0)), x.e_theta_im.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(3.0)), x.e_phi_re.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(4.0)), x.e_phi_im.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(5.0)), x.e_theta_re.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(6.0)), x.e_theta_im.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(7.0)), x.e_phi_re.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(8.0)), x.e_phi_im.slice(1), "absdiff", 2e-5));
    CHECK(arma::approx_equal(A, x.coupling_re, "absdiff", 1e-5));
    CHECK(arma::approx_equal(B, x.coupling_im, "absdiff", 1e-5));

    std::remove("test2.qdant");
}

TEST_CASE("Arrayant writing to QDANT file - Complex test")
{
    // Create new file
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

    ant.coupling_re = {1.0};
    ant.coupling_im = {0.1};
    ant.center_frequency = 2.0e9;
    ant.name = "name";

    // Append new antenna
    unsigned id = ant.qdant_write("test.qdant");

    CHECK(id == 2);

    arma::Mat<unsigned> layout, layout2;

    auto antI = quadriga_lib::qdant_read<float>("test.qdant", 1, &layout2);

    CHECK(antI.name == "bla");

    antI = quadriga_lib::qdant_read<float>("test.qdant", 2);
    CHECK(antI.name == "name");

    layout = {1, 2};
    CHECK(arma::all(arma::vectorise(layout2 - layout) == 0));

    id = ant.qdant_write("test.qdant", 112);
    CHECK(id == 112);

    // Layout can only contain valid entries (error expected)
    layout = {{1, 2, 112}, {112, 112, 6}};
    REQUIRE_THROWS_AS(ant.qdant_write("test.qdant", 5, layout), std::invalid_argument);

    id = ant.qdant_write("test.qdant", 6, layout);
    CHECK(id == 6);

    antI = quadriga_lib::qdant_read<float>("test.qdant", 6, &layout2);

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

// B-012: the writer stored magnitudes with a floor at -100 dB absolute power; samples below it read back as 0
// The file is accurate to 6e-8 of the element peak; an element at a low level is quantized relative to its own peak
TEST_CASE("Arrayant writing to QDANT file - Low-level pattern data is kept (B-012)")
{
    const std::string fn = "test_qdant_low_level.qdant";

    for (double scale : {1e-6, 1e-12, 1e-30})
    {
        std::remove(fn.c_str());
        auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.1, 10.0);
        a.e_theta_re *= scale;
        a.qdant_write(fn);
        auto b = quadriga_lib::qdant_read<double>(fn, 1);
        CHECK(qdant_err_element(a, b) < 1.2e-7);
        CHECK(arma::abs(arma::vectorise(b.e_theta_re)).max() > 0.5 * scale);
    }

    // One sample at about -110 dB (about -120 dB below the peak) and one exact zero in a normal pattern
    {
        std::remove(fn.c_str());
        auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.1, 10.0);
        a.e_theta_re(1, 0, 0) = 3e-6;
        a.e_theta_re(2, 0, 0) = 0.0;
        a.qdant_write(fn);
        auto b = quadriga_lib::qdant_read<double>(fn, 1);
        CHECK(qdant_err_element(a, b) < 1.2e-7);
        CHECK(std::abs(b.e_theta_re(1, 0, 0) / 3e-6 - 1.0) < 0.06); // kept, with at most 0.5 dB rounding
        CHECK(b.e_theta_re(2, 0, 0) == 0.0);
        CHECK(b.e_theta_im(2, 0, 0) == 0.0);
    }

    // Element 2 of 3 is zero everywhere, element 3 is at -200 dB (accurate relative to its own peak)
    {
        std::remove(fn.c_str());
        auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.1, 10.0);
        a.copy_element(0, arma::uvec{1, 2});
        a.e_theta_re.slice(1).zeros(), a.e_theta_im.slice(1).zeros(), a.e_phi_re.slice(1).zeros(), a.e_phi_im.slice(1).zeros();
        a.e_theta_re.slice(2) *= 1e-10;
        a.qdant_write(fn);
        auto b = quadriga_lib::qdant_read<double>(fn, 1);
        REQUIRE(b.n_elements() == 3);
        CHECK(arma::all(arma::vectorise(b.e_theta_re.slice(1)) == 0.0));
        CHECK(qdant_err_element(a, b) < 1.2e-7);
        CHECK(arma::abs(arma::vectorise(b.e_theta_re.slice(2))).max() > 0.5e-10);
    }

    // Float pattern at 1e-20 (power 1e-40 is below the smallest normal float); the float reader parses the dB values
    // (about -400) as float, whose resolution there is 3e-5 dB, i.e. 2e-6 of the amplitude
    {
        std::remove(fn.c_str());
        auto a = quadriga_lib::generate_arrayant_custom<float>(90.0f, 90.0f, 0.1f, 10.0f);
        a.e_theta_re *= 1e-20f;
        a.qdant_write(fn);
        auto b = quadriga_lib::qdant_read<float>(fn, 1);
        CHECK(qdant_err_element(a, b) < 1e-5);
    }

    std::remove(fn.c_str());
}

// B-038: element positions, coupling and pattern values were written with 6 significant digits
// The file has single-precision-equivalent accuracy: positions and coupling (magnitude, phase) as floats, pattern
// samples within 6e-8 of the element peak
TEST_CASE("Arrayant writing to QDANT file - Single-precision accuracy of positions, coupling and pattern (B-038)")
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
        a.coupling_re.randn(3, 2), a.coupling_im.randn(3, 2);
        a.coupling_re(0, 0) = 0.123456789;
        a.center_frequency = 3.456789012e9;
        a.qdant_write(fn);

        auto b = quadriga_lib::qdant_read<double>(fn, 1);

        // Positions exact as float: the value read back rounds to the same float (before: 6 significant digits, 123.457)
        arma::fmat pos_float = arma::conv_to<arma::fmat>::from(a.element_pos);
        CHECK(arma::approx_equal(arma::conv_to<arma::fmat>::from(b.element_pos), pos_float, "absdiff", 0.0f));
        CHECK((float)b.element_pos(0, 1) == 123.456789f);
        CHECK(std::abs(b.element_pos(1, 1) / 0.000123456789 - 1.0) < 1e-7);

        // Coupling from float magnitude and phase: each entry within 3e-7 of its magnitude
        arma::cx_mat ca(a.coupling_re, a.coupling_im), cb(b.coupling_re, b.coupling_im);
        CHECK(arma::max(arma::vectorise(arma::abs(cb - ca) / arma::abs(ca))) < 3e-7);

        // Pattern within 1.2e-7 of the element peak (float epsilon), also for the element at -120 dB; samples far
        // below the peak are not accurate to 1.2e-7 of their own amplitude
        CHECK(qdant_err_element(a, b) < 1.2e-7);
        CHECK(b.center_frequency == a.center_frequency);
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
        a.coupling_re.randn(2, 2), a.coupling_im.randn(2, 2);
        a.qdant_write(fn);

        auto b = quadriga_lib::qdant_read<float>(fn, 1);
        CHECK(arma::approx_equal(b.element_pos, a.element_pos, "absdiff", 0.0f));
        CHECK(qdant_rel_err(a.coupling_re, b.coupling_re) < 1e-6);
        CHECK(qdant_rel_err(a.coupling_im, b.coupling_im) < 1e-6);
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

    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.copy_element(0, {1, 2});
    a.element_pos = {{1200.0, 0.0001, 2.5e-7}, {-0.0, 1e-5, 1e15}, {123456.0, 0.1, 16777216.0}};
    std::remove(fn.c_str());
    a.qdant_write(fn);
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
        a.qdant_write(fn_loc);
        std::setlocale(LC_NUMERIC, old_locale.c_str());
        INFO("locale " << comma_locale);
        CHECK(read_text(fn_loc) == txt);
        std::remove(fn_loc.c_str());
    }
    else
    {
        std::setlocale(LC_NUMERIC, old_locale.c_str());
        WARN("No locale with a decimal comma installed; the locale part of this test was skipped.");
    }
    std::remove(fn.c_str());
}

// B-041 (documented behaviour): a write without a layout replaces the layout in the file by all IDs in file order
TEST_CASE("Arrayant writing to QDANT file - A write without layout resets the layout (B-041)")
{
    const std::string fn = "test_qdant_layout_reset.qdant";
    std::remove(fn.c_str());

    auto a = quadriga_lib::generate_arrayant_omni<double>(30.0);
    a.qdant_write(fn, 1), a.qdant_write(fn, 2), a.qdant_write(fn, 3);
    arma::u32_mat L = {{1, 2}, {3, 3}};
    a.qdant_write(fn, 3, L);

    arma::u32_mat L1;
    quadriga_lib::qdant_read<double>(fn, 1, &L1);
    REQUIRE(arma::size(L1) == arma::size(L));
    CHECK(arma::all(arma::vectorise(L1 == L)));

    // Append without layout
    CHECK(a.qdant_write(fn) == 4);
    quadriga_lib::qdant_read<double>(fn, 1, &L1);
    arma::u32_mat L_reset = {{1, 2, 3, 4}};
    REQUIRE(arma::size(L1) == arma::size(L_reset));
    CHECK(arma::all(arma::vectorise(L1 == L_reset)));

    // Overwriting an entry without layout also resets a custom layout
    a.qdant_write(fn, 4, L);
    a.qdant_write(fn, 2);
    quadriga_lib::qdant_read<double>(fn, 1, &L1);
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

    auto ant = quadriga_lib::generate_arrayant_xpol<double>(10.0);
    ant.name = "second";
    unsigned id = ant.qdant_write(fn); // appended with ID 2
    CHECK(id == 2);

    auto first = quadriga_lib::qdant_read<double>(fn, 1);
    CHECK(first.name == "first");
    auto second = quadriga_lib::qdant_read<double>(fn, 2);
    CHECK(second.name == "second");
    CHECK(second.e_theta_re.n_slices == 2);
    CHECK(arma::approx_equal(second.e_theta_re, ant.e_theta_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(second.e_phi_re, ant.e_phi_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(second.azimuth_grid, ant.azimuth_grid, "absdiff", 1e-6));

    // Overwrite entry 1 in place
    ant.name = "replaced";
    CHECK(ant.qdant_write(fn, 1) == 1);
    CHECK(quadriga_lib::qdant_read<double>(fn, 1).name == "replaced");
    CHECK(quadriga_lib::qdant_read<double>(fn, 2).name == "second");

    // Every node in the file carries the prefix
    std::ifstream f(fn);
    std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(txt.find("<name>") == std::string::npos);
    CHECK(txt.find("<arrayant") == std::string::npos);
    CHECK(txt.find("<EthetaMag") == std::string::npos);
    std::remove(fn.c_str());
}
