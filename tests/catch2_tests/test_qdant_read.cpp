// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// qdant_read: hand-written QDANT files (magnitudes in dB, phases and grids in degrees) give the expected linear
// fields, radians, coupling and center frequency; also XML namespaces, the ID lookup and the errors for missing IDs,
// malformed counts and empty grids. Writing and multi-entry files are covered by test_qdant_write.

#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"

#include <iostream>
#include <fstream>
#include <string>

TEST_CASE("Arrayant reading from QDANT file - Minimal test")
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

    float pi = arma::datum::pi;
    auto x = quadriga_lib::qdant_read<float>("test.qdant", {1})[0];

    arma::fmat A = 20 * log10(x.e_theta_re.slice(0));
    arma::fmat B = arma::reshape(arma::linspace<arma::fvec>(1, 15, 15), 3, 5).t();
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-6));

    B.zeros();
    CHECK(arma::approx_equal(x.e_theta_im.slice(0), B, "absdiff", 1e-6));
    CHECK(arma::approx_equal(x.e_phi_re.slice(0), B, "absdiff", 1e-6));
    CHECK(arma::approx_equal(x.e_phi_im.slice(0), B, "absdiff", 1e-6));

    B = {-pi, 0.0f, pi / 2.0f};
    CHECK(arma::approx_equal(x.azimuth_grid, B.t(), "absdiff", 1e-6));

    B = {-pi / 2.0f, -pi / 4.0f, 0.0f, pi / 4.0f, pi / 2.0f};
    CHECK(arma::approx_equal(x.elevation_grid, B.t(), "absdiff", 1e-6));

    B = {0.0f, 0.0f, 0.0f};
    CHECK(arma::approx_equal(x.element_pos, B.t(), "absdiff", 1e-6));

    B = {1.0f};
    CHECK(arma::approx_equal(x.port_weights_re, B, "absdiff", 1e-6));

    B = {0.0f};
    CHECK(arma::approx_equal(x.port_weights_im, B, "absdiff", 1e-6));
    CHECK(std::abs(x.center_freq - 299792447.0f) < 0.1f);

    std::remove("test.qdant");
}

TEST_CASE("Arrayant reading from QDANT file - Multi-Element Array")
{
    std::ofstream f;
    f.open("test2.qdant");
    f << "<qdant><arrayant>" << std::endl;
    f << "<name>bla</name>" << std::endl;
    f << "<NoElements>2</NoElements >" << std::endl;
    f << "<ElevationGrid>-90 -45 0 45 90</ElevationGrid>" << std::endl;
    f << "<AzimuthGrid>-180 0 90</AzimuthGrid>" << std::endl;
    f << "<EthetaMag   el=\"1\">1 2 3 4 5 6 7 8 9 10 11 12 13 14 15</EthetaMag>" << std::endl;
    f << "<EphiMag     el=\"1\">2 3 4 5 6 7 8 9 10 11 12 13 14 15 16</EphiMag>" << std::endl;
    f << "<EthetaMag   el=\"2\">3 4 5 6 7 8 9 10 11 12 13 14 15 16 17</EthetaMag>" << std::endl;
    f << "<EphiMag     el=\"2\">4 5 6 7 8 9 10 11 12 13 14 15 16 17 18</EphiMag>" << std::endl;
    f << "<EthetaPhase el=\"1\">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</EthetaPhase>" << std::endl;
    f << "<EphiPhase   el=\"1\">2 2 2 2 2 2 2 2 2 2 2 2 2 2 2</EphiPhase>" << std::endl;
    f << "<EthetaPhase el=\"2\">3 3 3 3 3 3 3 3 3 3 3 3 3 3 3</EthetaPhase>" << std::endl;
    f << "<EphiPhase   el=\"2\">4 4 4 4 4 4 4 4 4 4 4 4 4 4 4</EphiPhase>" << std::endl;
    f << "<CouplingAbs>1 2 3 4 5 6</CouplingAbs>" << std::endl;
    f << "<CouplingPhase>5 10 15 20 25 30</CouplingPhase>" << std::endl;
    f << "</arrayant></qdant>" << std::endl;
    f.close();

    double pi = arma::datum::pi;
    auto x = quadriga_lib::qdant_read<double>("test2.qdant", {1})[0];

    arma::mat A = 20 * log10(sqrt(x.e_theta_re.slice(0) % x.e_theta_re.slice(0) + x.e_theta_im.slice(0) % x.e_theta_im.slice(0)));
    arma::mat B = arma::reshape(arma::linspace<arma::vec>(1, 15, 15), 3, 5).t();
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = 20 * log10(sqrt(x.e_phi_re.slice(0) % x.e_phi_re.slice(0) + x.e_phi_im.slice(0) % x.e_phi_im.slice(0)));
    B = arma::reshape(arma::linspace<arma::vec>(2, 16, 15), 3, 5).t();
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = 20 * log10(sqrt(x.e_theta_re.slice(1) % x.e_theta_re.slice(1) + x.e_theta_im.slice(1) % x.e_theta_im.slice(1)));
    B = arma::reshape(arma::linspace<arma::vec>(3, 17, 15), 3, 5).t();
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = 20 * log10(sqrt(x.e_phi_re.slice(1) % x.e_phi_re.slice(1) + x.e_phi_im.slice(1) % x.e_phi_im.slice(1)));
    B = arma::reshape(arma::linspace<arma::vec>(4, 18, 15), 3, 5).t();
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = arma::atan2(x.e_theta_im.slice(0), x.e_theta_re.slice(0));
    B.fill(pi / 180.0);
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = arma::atan2(x.e_phi_im.slice(0), x.e_phi_re.slice(0));
    B.fill(2.0 * pi / 180.0);
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = arma::atan2(x.e_theta_im.slice(1), x.e_theta_re.slice(1));
    B.fill(3.0 * pi / 180.0);
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = arma::atan2(x.e_phi_im.slice(1), x.e_phi_re.slice(1));
    B.fill(4.0 * pi / 180.0);
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = sqrt(x.port_weights_re % x.port_weights_re + x.port_weights_im % x.port_weights_im);
    B = arma::reshape(arma::linspace<arma::vec>(1, 6, 6), 2, 3);
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = arma::atan2(x.port_weights_im, x.port_weights_re) * 180.0 / pi;
    B = arma::reshape(arma::linspace<arma::vec>(5, 30, 6), 2, 3);
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-12));

    std::remove("test2.qdant");
}

TEST_CASE("Arrayant reading from QDANT file - More complex test")
{
    std::ofstream f;
    f.open("test.qdant");
    f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?><qdant xmlns:xx=\"test\">" << std::endl;
    f << "<xx:layout>1,1 1,1 1,1</xx:layout>" << std::endl;
    f << "<xx:arrayant id=\"1\">" << std::endl;
    f << "<xx:AzimuthGrid>-90 -45 0 45 90</xx:AzimuthGrid>" << std::endl;
    f << "<xx:ElevationGrid>-90 0 90</xx:ElevationGrid>" << std::endl;
    f << "<xx:EphiMag>0  0  0  0  0  0  0  0  0  0  0  0  0  0  0</xx:EphiMag>" << std::endl;
    f << "<xx:EphiPhase>90  90  90  90  90  90  90  90  90  90  90  90  90  90  90</xx:EphiPhase>" << std::endl;
    f << "<xx:EthetaMag>3  3  3  3  3  3  3  3  3  3  3  3  3  3  3</xx:EthetaMag>" << std::endl;
    f << "<xx:EthetaPhase>-90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90</xx:EthetaPhase>" << std::endl;
    f << "<xx:ElementPosition>1,2,3</xx:ElementPosition>" << std::endl;
    f << "<xx:CouplingAbs>1</xx:CouplingAbs>" << std::endl;
    f << "<xx:CouplingPhase>45</xx:CouplingPhase>" << std::endl;
    f << "<xx:CenterFrequency>3e9</xx:CenterFrequency>" << std::endl;
    f << "</xx:arrayant></qdant>" << std::endl;
    f.close();

    double pi = arma::datum::pi;

    arma::Mat<unsigned> layout;
    auto x = quadriga_lib::qdant_read<double>("test.qdant", {1}, &layout)[0];

    arma::mat B = {-pi / 2.0, -pi / 4.0, 0.0, pi / 4.0, pi / 2.0};
    CHECK(arma::approx_equal(x.azimuth_grid, B.t(), "absdiff", 1e-13));

    B = {-pi / 2.0, 0.0, pi / 2.0};
    CHECK(arma::approx_equal(x.elevation_grid, B.t(), "absdiff", 1e-13));

    B.zeros(3, 5);
    CHECK(arma::approx_equal(x.e_theta_re.slice(0), B, "absdiff", 1e-13));
    CHECK(arma::approx_equal(x.e_phi_re.slice(0), B, "absdiff", 1e-13));

    B.fill(-std::sqrt(std::pow(10, 0.3)));
    CHECK(arma::approx_equal(x.e_theta_im.slice(0), B, "absdiff", 1e-13));

    B.ones();
    CHECK(arma::approx_equal(x.e_phi_im.slice(0), B, "absdiff", 1e-13));

    B = {1.0, 2.0, 3.0};
    CHECK(arma::approx_equal(x.element_pos, B.t(), "absdiff", 1e-13));

    B = {1.0 / std::sqrt(2.0)};
    CHECK(arma::approx_equal(x.port_weights_re, B, "absdiff", 1e-13));
    CHECK(arma::approx_equal(x.port_weights_im, B, "absdiff", 1e-13));

    CHECK(std::abs(x.center_freq - 3.0e9) < 0.1f);
    CHECK(x.name == "unknown");
    CHECK(arma::all(arma::vectorise(layout) == 1));

    std::remove("test.qdant");
}
// A requested ID that is not in the file throws, also ID 1 when the first entry has another ID
TEST_CASE("Arrayant reading from QDANT file - Missing ID throws")
{
    const std::string fn = "test_qdant_missing_id.qdant";
    std::remove(fn.c_str());

    auto a = quadriga_lib::generate_antenna<double>("omni", 10.0);
    a.name = "two";
    quadriga_lib::qdant_write(fn, a, 2);
    a.name = "three";
    quadriga_lib::qdant_write(fn, a, 3);

    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, {1}), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::qdant_read<float>(fn, {4}), std::invalid_argument);
    CHECK(quadriga_lib::qdant_read<double>(fn, {2})[0].name == "two");
    CHECK(quadriga_lib::qdant_read<double>(fn, {3})[0].name == "three");

    // Without ids, qdant_read takes the IDs from the layout and is not affected
    arma::u32_mat layout;
    auto v = quadriga_lib::qdant_read<double>(fn, {}, &layout);
    REQUIRE(v.size() == 2);
    CHECK(v[0].name == "two");
    CHECK(v[1].name == "three");

    // A first entry without an "id" attribute is entry 1; an entry with another ID is not
    std::ofstream f(fn);
    f << "<qdant><arrayant><name>no-id</name><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant>"
      << "<arrayant id=\"2\"><name>id-2</name><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant></qdant>\n";
    f.close();
    CHECK(quadriga_lib::qdant_read<double>(fn, {1})[0].name == "no-id");
    CHECK(quadriga_lib::qdant_read<double>(fn, {2})[0].name == "id-2");

    f.open(fn);
    f << "<qdant><arrayant id=\"2\"><name>id-2</name><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant>"
      << "<arrayant><name>no-id</name><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant></qdant>\n";
    f.close();
    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, {1}), std::invalid_argument);

    // A missing file is a file problem, a missing ID an argument error
    std::remove(fn.c_str());
    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, {1}), std::runtime_error);
}

// A list of IDs is returned in the given order, duplicates included; an ID of 0 or one not in the file throws
TEST_CASE("Arrayant reading from QDANT file - List of IDs")
{
    const std::string fn = "test_qdant_id_list.qdant";
    std::remove(fn.c_str());

    std::vector<quadriga_lib::arrayant<double>> v;
    const char *names[3] = {"one", "two", "three"};
    for (int i = 0; i < 3; ++i)
    {
        v.push_back(quadriga_lib::generate_antenna<double>("omni", 10.0 + 5.0 * i));
        v.back().name = names[i];
        v.back().center_freq = 1.0e9 * (i + 1);
    }
    quadriga_lib::qdant_write(fn, v);

    arma::u32_mat layout;
    auto a = quadriga_lib::qdant_read<double>(fn, {3, 1, 3}, &layout);
    REQUIRE(a.size() == 3);
    CHECK(a[0].name == "three");
    CHECK(a[1].name == "one");
    CHECK(a[2].name == "three");
    CHECK(a[0].center_freq == 3.0e9);
    CHECK(a[1].center_freq == 1.0e9);
    CHECK(a[0].azimuth_grid.n_elem == a[2].azimuth_grid.n_elem);
    CHECK(arma::approx_equal(a[0].e_theta_re, a[2].e_theta_re, "absdiff", 0.0));
    CHECK(arma::approx_equal(arma::conv_to<arma::vec>::from(arma::vectorise(layout)), arma::vec{1.0, 2.0, 3.0}, "absdiff", 0.0));

    // No IDs: all entries; the same as listing them
    auto all = quadriga_lib::qdant_read<float>(fn);
    auto lst = quadriga_lib::qdant_read<float>(fn, arma::uvec{1, 2, 3});
    REQUIRE(all.size() == 3);
    REQUIRE(lst.size() == 3);
    for (size_t i = 0; i < 3; ++i)
        CHECK(all[i].name == lst[i].name);

    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, {0}), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, {4}), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, {1, 2, 4}), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, {1, 0}), std::invalid_argument);

    std::remove(fn.c_str());
}

// Writes the given text to a file and returns the exception type of qdant_read (or "none")
static std::string qdant_read_exception(const std::string &fn, const std::string &text, unsigned id = 1)
{
    std::ofstream f(fn);
    f << text;
    f.close();
    try
    {
        quadriga_lib::qdant_read<double>(fn, {id});
        quadriga_lib::qdant_read<float>(fn);
    }
    catch (const std::invalid_argument &)
    {
        return "invalid_argument";
    }
    catch (const std::runtime_error &)
    {
        return "runtime_error";
    }
    catch (const std::bad_alloc &)
    {
        return "bad_alloc";
    }
    catch (const std::exception &e)
    {
        return std::string("other: ") + e.what();
    }
    return "none";
}

// Counts in the file that the data does not back throw instead of allocating memory
TEST_CASE("Arrayant reading from QDANT file - Malformed counts throw runtime_error")
{
    const std::string fn = "test_qdant_malformed.qdant";
    const std::string grid = "<ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid>";
    auto ant = [&](const std::string &body)
    { return "<qdant><arrayant id=\"1\">" + body + "</arrayant></qdant>\n"; };

    // 2e8 elements, no data
    CHECK(qdant_read_exception(fn, ant("<NoElements>200000000</NoElements>" + grid)) == "runtime_error");

    // Element count not backed by positions, coupling or pattern data
    CHECK(qdant_read_exception(fn, ant("<NoElements>2</NoElements>" + grid)) == "runtime_error");

    // Invalid element counts
    for (std::string n : {"0", "-1", "abc", "1e3", "2.5", "99999999999999999999999"})
        CHECK(qdant_read_exception(fn, ant("<NoElements>" + n + "</NoElements>" + grid)) == "runtime_error");

    // Element positions that do not match the element count
    CHECK(qdant_read_exception(fn, ant("<NoElements>3</NoElements><ElementPosition>0,0,0 1,1,1</ElementPosition>" + grid)) == "runtime_error");

    // Coupling without entries, or with a number of entries that is not a multiple of the element count
    CHECK(qdant_read_exception(fn, ant("<NoElements>2</NoElements><CouplingAbs> </CouplingAbs>" + grid)) == "runtime_error");
    CHECK(qdant_read_exception(fn, ant("<NoElements>2</NoElements><CouplingAbs>1,0,0</CouplingAbs>" + grid)) == "runtime_error");

    // Element count backed only by a pattern node whose text is too short for the grid
    CHECK(qdant_read_exception(fn, ant("<NoElements>3</NoElements><ElevationGrid>-90 0 90</ElevationGrid>"
                                       "<AzimuthGrid>-90 0 90</AzimuthGrid><EthetaMag el=\"3\">0</EthetaMag>")) == "runtime_error");

    // Many elements without pattern data on a fine grid (positions present, 300 x 181 x 361 zero samples)
    {
        std::string pos;
        for (int i = 0; i < 300; ++i)
            pos += i == 0 ? "0,0,0" : " 0,0,0";
        std::string el_grid, az_grid;
        for (int i = -90; i <= 90; ++i)
            el_grid += std::to_string(i) + " ";
        for (int i = -180; i <= 180; ++i)
            az_grid += std::to_string(i) + " ";
        std::string cpl;
        for (int i = 0; i < 300; ++i)
            cpl += i == 0 ? "1" : " 1";
        CHECK(qdant_read_exception(fn, ant("<NoElements>300</NoElements><ElementPosition>" + pos + "</ElementPosition>" +
                                           "<CouplingAbs>" + cpl + "</CouplingAbs>"
                                           "<ElevationGrid>" + el_grid + "</ElevationGrid><AzimuthGrid>" + az_grid + "</AzimuthGrid>")) == "runtime_error");

        // 5000 elements without coupling: the identity coupling alone would have 25 million entries
        pos.clear();
        for (int i = 0; i < 5000; ++i)
            pos += i == 0 ? "0,0,0" : " 0,0,0";
        CHECK(qdant_read_exception(fn, ant("<NoElements>5000</NoElements><ElementPosition>" + pos + "</ElementPosition>" + grid)) == "runtime_error");
    }

    // Malformed number lists (Armadillo reports them as std::logic_error)
    CHECK(qdant_read_exception(fn, ant("<ElevationGrid>0 1;2</ElevationGrid><AzimuthGrid>0</AzimuthGrid>")) == "runtime_error");
    CHECK(qdant_read_exception(fn, ant(grid + "<EthetaMag>1 2;3</EthetaMag>")) == "runtime_error");
    CHECK(qdant_read_exception(fn, ant("<ElementPosition>0,0;0</ElementPosition>" + grid)) == "runtime_error");

    // Malformed layout and IDs
    CHECK(qdant_read_exception(fn, "<qdant><layout>1,2 3</layout><arrayant id=\"1\">" + grid + "</arrayant></qdant>\n") == "runtime_error");
    CHECK(qdant_read_exception(fn, "<qdant><arrayant id=\"abc\">" + grid + "</arrayant></qdant>\n", 2) == "runtime_error");
    CHECK(qdant_read_exception(fn, "<qdant><arrayant id=\"99999999999\">" + grid + "</arrayant></qdant>\n", 2) == "runtime_error");

    // Valid files: an element count backed by positions only (elements without pattern data read as zeros)
    {
        std::ofstream f(fn);
        f << ant("<NoElements>2</NoElements><ElementPosition>0,0,0 1,2,3</ElementPosition>"
                 "<ElevationGrid>-90 0 90</ElevationGrid><AzimuthGrid>-90 0 90</AzimuthGrid>"
                 "<EthetaMag el=\"1\">0 0 0 0 0 0 0 0 0</EthetaMag>");
        f.close();
        auto a = quadriga_lib::qdant_read<double>(fn, {1})[0];
        CHECK(a.n_elements() == 2);
        CHECK(arma::all(arma::vectorise(a.e_theta_re.slice(0)) == 1.0));
        CHECK(arma::all(arma::vectorise(a.e_theta_re.slice(1)) == 0.0));
        CHECK(a.element_pos(2, 1) == 3.0);
    }

    // ... or by pattern nodes only
    {
        std::ofstream f(fn);
        f << ant("<NoElements>2</NoElements><ElevationGrid>0</ElevationGrid><AzimuthGrid>-90 90</AzimuthGrid>"
                 "<EthetaMag el=\"1\">0 0</EthetaMag><EphiMag el=\"2\">0 -inf</EphiMag>");
        f.close();
        auto a = quadriga_lib::qdant_read<double>(fn, {1})[0];
        CHECK(a.n_elements() == 2);
        CHECK(a.e_phi_re(0, 0, 1) == 1.0);
        CHECK(a.e_phi_re(0, 1, 1) == 0.0);
    }

    std::remove(fn.c_str());
}

TEST_CASE("Arrayant reading from QDANT file - Values without data are limited per read")
{
    // Entry 1 has one value without data and entry 2 has 2^24, so every read stops before entry 2 allocates
    const std::string fn = "test_qdant_read_limit.qdant";
    const std::string small = "<elevation_grid>0</elevation_grid><azimuth_grid>0</azimuth_grid>";
    const std::string weights = "<n_elements>4096</n_elements><e_theta_mag el=\"4096\">0</e_theta_mag>"; // No grid: throws after the limit
    std::string patterns = "<n_elements>256</n_elements><element_pos>0,0,0", grid = "0";
    for (int i = 1; i < 256; ++i)
        patterns += " 0,0,0", grid += " 0";
    patterns += "</element_pos><elevation_grid>" + grid + "</elevation_grid><azimuth_grid>" + grid + "</azimuth_grid>";

    auto read_error = [&](const std::string &entry, const arma::uvec &ids = {}, const std::string &padding = "") -> std::string
    {
        {
            std::ofstream f(fn, std::ios::binary);
            f << "<qdant><!--" << padding << "--><layout>1 2</layout><arrayant id=\"1\">" << small << "</arrayant><arrayant id=\"2\">" << entry << "</arrayant></qdant>";
        }
        try
        {
            quadriga_lib::qdant_read<float>(fn, ids);
        }
        catch (const std::runtime_error &e)
        {
            return e.what();
        }
        return "";
    };

    CHECK_THAT(read_error(patterns), Catch::Matchers::ContainsSubstring("too many values without data"));
    CHECK_THAT(read_error(weights), Catch::Matchers::ContainsSubstring("too many values without data"));
    CHECK_THAT(read_error(weights, {2}), Catch::Matchers::ContainsSubstring("must have an 'elevation_grid'")); // One entry fits
    CHECK_THAT(read_error(weights, {}, std::string(132000, ' ')), Catch::Matchers::ContainsSubstring("must have an 'elevation_grid'")); // 128 values per file byte

    std::remove(fn.c_str());
}

TEST_CASE("Arrayant reading from QDANT file - Empty grid throws")
{
    // An empty grid with pattern nodes present throws an error that names the grid
    const std::string fn = "test_qdant_empty_grid.qdant";
    for (int empty_az = 0; empty_az < 2; ++empty_az)
    {
        std::ofstream f(fn);
        f << "<?xml version=\"1.0\"?>\n<qdant xmlns=\"http://www.quadriga-channel-model.de\">\n<layout>1</layout>\n"
          << "<arrayant id=\"1\">\n<name>e</name>\n<CenterFrequency>299792458</CenterFrequency>\n"
          << (empty_az ? "<ElevationGrid>0</ElevationGrid>\n<AzimuthGrid></AzimuthGrid>\n" : "<ElevationGrid></ElevationGrid>\n<AzimuthGrid>0</AzimuthGrid>\n")
          << "<EthetaMag el=\"1\"></EthetaMag>\n<EthetaPhase el=\"1\"></EthetaPhase>\n<EphiMag el=\"1\"></EphiMag>\n"
          << "</arrayant>\n</qdant>\n";
        f.close();
        const std::string grid = empty_az ? "azimuth_grid" : "elevation_grid";
        CHECK_THROWS_WITH(quadriga_lib::qdant_read<double>(fn, {1}), Catch::Matchers::ContainsSubstring(grid));
        CHECK_THROWS_WITH(quadriga_lib::qdant_read<float>(fn, {1}), Catch::Matchers::ContainsSubstring(grid));
    }
    std::remove(fn.c_str());
}

TEST_CASE("Arrayant reading from QDANT file - All entries follow the layout")
{
    const std::string fn = "test_qdant_layout_order.qdant";
    std::remove(fn.c_str());
    for (unsigned i = 1; i <= 3; ++i)
    {
        auto ant = quadriga_lib::generate_antenna<double>("omni", 30.0);
        ant.center_freq = 1.0e9 * i;
        quadriga_lib::qdant_write(fn, ant, i);
    }
    auto ant = quadriga_lib::generate_antenna<double>("omni", 30.0);
    ant.center_freq = 2.0e9;
    quadriga_lib::qdant_write(fn, ant, 2, arma::u32_mat({{2, 1}, {2, 3}})); // Column-major: 2, 2, 1, 3

    auto all = quadriga_lib::qdant_read<double>(fn);
    REQUIRE(all.size() == 3);
    CHECK(all[0].center_freq == 2.0e9);
    CHECK(all[1].center_freq == 1.0e9);
    CHECK(all[2].center_freq == 3.0e9);
    std::remove(fn.c_str());
}

TEST_CASE("Arrayant reading from QDANT file - Malformed layout values")
{
    const std::string fn = "test_qdant_bad_layout.qdant";
    std::remove(fn.c_str());
    auto ant = quadriga_lib::generate_antenna<double>("omni", 30.0);
    quadriga_lib::qdant_write(fn, ant, 1);
    quadriga_lib::qdant_write(fn, ant, 2);

    std::ifstream in(fn);
    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();
    const std::string text = buffer.str();
    const size_t first = text.find("<layout>") + 8, last = text.find("</layout>");
    REQUIRE(last != std::string::npos);
    const auto with_layout = [&](const std::string &layout)
    {
        std::ofstream out(fn, std::ios::trunc);
        out << text.substr(0, first) << layout << text.substr(last);
    };

    for (const char *bad : {"1.5", "-1", "abc", "99999999999", "1 x", "1e0"})
    {
        with_layout(bad);
        CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn), std::runtime_error);
        CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, {1}), std::runtime_error);
    }

    with_layout("0 2"); // Zeros stay allowed and are skipped
    CHECK(quadriga_lib::qdant_read<double>(fn).size() == 1);
    with_layout("1,2 2,1"); // Two columns
    arma::u32_mat layout;
    CHECK(quadriga_lib::qdant_read<double>(fn, {}, &layout).size() == 2);
    CHECK(layout.n_rows == 2);
    CHECK(layout.n_cols == 2);
    std::remove(fn.c_str());
}
