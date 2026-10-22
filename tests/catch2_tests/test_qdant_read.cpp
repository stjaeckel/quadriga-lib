// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

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
    auto x = quadriga_lib::qdant_read<float>("test.qdant");

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
    CHECK(arma::approx_equal(x.coupling_re, B, "absdiff", 1e-6));

    B = {0.0f};
    CHECK(arma::approx_equal(x.coupling_im, B, "absdiff", 1e-6));
    CHECK(std::abs(x.center_frequency - 299792447.0f) < 0.1f);

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
    auto x = quadriga_lib::qdant_read<double>("test2.qdant");

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

    A = sqrt(x.coupling_re % x.coupling_re + x.coupling_im % x.coupling_im);
    B = arma::reshape(arma::linspace<arma::vec>(1, 6, 6), 2, 3);
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-13));

    A = arma::atan2(x.coupling_im, x.coupling_re) * 180.0 / pi;
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
    auto x = quadriga_lib::qdant_read<double>("test.qdant", 1, &layout);

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
    CHECK(arma::approx_equal(x.coupling_re, B, "absdiff", 1e-13));
    CHECK(arma::approx_equal(x.coupling_im, B, "absdiff", 1e-13));

    CHECK(std::abs(x.center_frequency - 3.0e9) < 0.1f);
    CHECK(x.name == "unknown");
    CHECK(arma::all(arma::vectorise(layout) == 1));

    std::remove("test.qdant");
}
// A request for ID 1 returned the first entry of the file, also when that entry has another ID
TEST_CASE("Arrayant reading from QDANT file - Missing ID throws")
{
    const std::string fn = "test_qdant_missing_id.qdant";
    std::remove(fn.c_str());

    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.name = "two";
    a.qdant_write(fn, 2);
    a.name = "three";
    a.qdant_write(fn, 3);

    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, 1), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::qdant_read<float>(fn, 4), std::invalid_argument);
    CHECK(quadriga_lib::qdant_read<double>(fn, 2).name == "two");
    CHECK(quadriga_lib::qdant_read<double>(fn, 3).name == "three");

    // qdant_read_multi takes the IDs from the layout and is not affected
    arma::u32_mat layout;
    auto v = quadriga_lib::qdant_read_multi<double>(fn, &layout);
    REQUIRE(v.size() == 2);
    CHECK(v[0].name == "two");
    CHECK(v[1].name == "three");

    // A first entry without an "id" attribute is entry 1; an entry with another ID is not
    std::ofstream f(fn);
    f << "<qdant><arrayant><name>no-id</name><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant>"
      << "<arrayant id=\"2\"><name>id-2</name><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant></qdant>\n";
    f.close();
    CHECK(quadriga_lib::qdant_read<double>(fn, 1).name == "no-id");
    CHECK(quadriga_lib::qdant_read<double>(fn, 2).name == "id-2");

    f.open(fn);
    f << "<qdant><arrayant id=\"2\"><name>id-2</name><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant>"
      << "<arrayant><name>no-id</name><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant></qdant>\n";
    f.close();
    CHECK_THROWS_AS(quadriga_lib::qdant_read<double>(fn, 1), std::invalid_argument);

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
        quadriga_lib::qdant_read<double>(fn, id);
        quadriga_lib::qdant_read_multi<float>(fn);
    }
    catch (const std::invalid_argument &)
    {
        return "invalid_argument";
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

// The reader allocated memory from counts in the file without checking them against the data
TEST_CASE("Arrayant reading from QDANT file - Malformed counts throw invalid_argument")
{
    const std::string fn = "test_qdant_malformed.qdant";
    const std::string grid = "<ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid>";
    auto ant = [&](const std::string &body)
    { return "<qdant><arrayant id=\"1\">" + body + "</arrayant></qdant>\n"; };

    // The 138-byte file of the report: 2e8 elements, no data
    CHECK(qdant_read_exception(fn, ant("<NoElements>200000000</NoElements>" + grid)) == "invalid_argument");

    // Element count not backed by positions, coupling or pattern data
    CHECK(qdant_read_exception(fn, ant("<NoElements>2</NoElements>" + grid)) == "invalid_argument");

    // Invalid element counts
    for (std::string n : {"0", "-1", "abc", "1e3", "2.5", "99999999999999999999999"})
        CHECK(qdant_read_exception(fn, ant("<NoElements>" + n + "</NoElements>" + grid)) == "invalid_argument");

    // Element positions that do not match the element count
    CHECK(qdant_read_exception(fn, ant("<NoElements>3</NoElements><ElementPosition>0,0,0 1,1,1</ElementPosition>" + grid)) == "invalid_argument");

    // Coupling without entries, or with a number of entries that is not a multiple of the element count
    CHECK(qdant_read_exception(fn, ant("<NoElements>2</NoElements><CouplingAbs> </CouplingAbs>" + grid)) == "invalid_argument");
    CHECK(qdant_read_exception(fn, ant("<NoElements>2</NoElements><CouplingAbs>1,0,0</CouplingAbs>" + grid)) == "invalid_argument");

    // Element count backed only by a pattern node whose text is too short for the grid
    CHECK(qdant_read_exception(fn, ant("<NoElements>3</NoElements><ElevationGrid>-90 0 90</ElevationGrid>"
                                       "<AzimuthGrid>-90 0 90</AzimuthGrid><EthetaMag el=\"3\">0</EthetaMag>")) == "invalid_argument");

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
                                           "<ElevationGrid>" + el_grid + "</ElevationGrid><AzimuthGrid>" + az_grid + "</AzimuthGrid>")) == "invalid_argument");

        // 5000 elements without coupling: the identity coupling alone would have 25 million entries
        pos.clear();
        for (int i = 0; i < 5000; ++i)
            pos += i == 0 ? "0,0,0" : " 0,0,0";
        CHECK(qdant_read_exception(fn, ant("<NoElements>5000</NoElements><ElementPosition>" + pos + "</ElementPosition>" + grid)) == "invalid_argument");
    }

    // Malformed number lists (Armadillo reports them as std::logic_error)
    CHECK(qdant_read_exception(fn, ant("<ElevationGrid>0 1;2</ElevationGrid><AzimuthGrid>0</AzimuthGrid>")) == "invalid_argument");
    CHECK(qdant_read_exception(fn, ant(grid + "<EthetaMag>1 2;3</EthetaMag>")) == "invalid_argument");
    CHECK(qdant_read_exception(fn, ant("<ElementPosition>0,0;0</ElementPosition>" + grid)) == "invalid_argument");

    // Malformed layout and IDs
    CHECK(qdant_read_exception(fn, "<qdant><layout>1,2 3</layout><arrayant id=\"1\">" + grid + "</arrayant></qdant>\n") == "invalid_argument");
    CHECK(qdant_read_exception(fn, "<qdant><arrayant id=\"abc\">" + grid + "</arrayant></qdant>\n", 2) == "invalid_argument");
    CHECK(qdant_read_exception(fn, "<qdant><arrayant id=\"99999999999\">" + grid + "</arrayant></qdant>\n", 2) == "invalid_argument");

    // Valid files: an element count backed by positions only (elements without pattern data read as zeros)
    {
        std::ofstream f(fn);
        f << ant("<NoElements>2</NoElements><ElementPosition>0,0,0 1,2,3</ElementPosition>"
                 "<ElevationGrid>-90 0 90</ElevationGrid><AzimuthGrid>-90 0 90</AzimuthGrid>"
                 "<EthetaMag el=\"1\">0 0 0 0 0 0 0 0 0</EthetaMag>");
        f.close();
        auto a = quadriga_lib::qdant_read<double>(fn);
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
        auto a = quadriga_lib::qdant_read<double>(fn);
        CHECK(a.n_elements() == 2);
        CHECK(a.e_phi_re(0, 0, 1) == 1.0);
        CHECK(a.e_phi_re(0, 1, 1) == 0.0);
    }

    std::remove(fn.c_str());
}
