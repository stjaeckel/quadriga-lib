// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"
#include "lodepng.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

TEST_CASE("Write PNG")
{
    const arma::uword N = 500;  // size along one edge
    const double maxVal = 10.0; // value at the upper‑right corner
    arma::mat gradient(N, N, arma::fill::none);
    const double scale = maxVal / double(N); // (= 10 / 98 for N = 50)

    for (arma::uword r = 0; r < N; ++r)
        for (arma::uword c = 0; c < N; ++c)
            gradient(r, c) = scale * c;

    quadriga_lib::write_png(gradient, "test.png");

    REQUIRE(std::filesystem::exists("test.png"));

    std::remove("test.png");
}
// Write "data" with write_png, decode the file and return the colormap row of each pixel (column-major like "data");
// -1 for a colour that is not in the map
static std::vector<int> png_colour_rows(const arma::mat &data, double mi = NAN, double ma = NAN, bool lg = false)
{
    const std::string fn = "test_png_range.png";
    quadriga_lib::write_png<double>(data, fn, "jet", mi, ma, lg);
    std::vector<unsigned char> img;
    unsigned w = 0, h = 0;
    unsigned err = lodepng::decode(img, w, h, fn);
    std::remove(fn.c_str());
    REQUIRE(err == 0);
    REQUIRE(w == data.n_cols);
    REQUIRE(h == data.n_rows);

    auto cmap = quadriga_lib::colormap("jet", true);
    std::vector<int> rows;
    for (arma::uword c = 0; c < data.n_cols; ++c)
        for (arma::uword r = 0; r < data.n_rows; ++r)
        {
            const unsigned char *px = &img[4 * (r * w + c)];
            int found = -1;
            for (arma::uword k = 0; k < cmap.n_rows && found < 0; ++k)
                if (cmap(k, 0) == px[0] && cmap(k, 1) == px[1] && cmap(k, 2) == px[2])
                    found = (int)k;
            rows.push_back(found);
        }
    return rows;
}

TEST_CASE("Write PNG - Degenerate ranges / NaN / Inf")
{
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    const int last = (int)quadriga_lib::colormap("jet", true).n_rows - 1;
    auto all_first = [](const std::vector<int> &v)
    { return std::all_of(v.begin(), v.end(), [](int x) { return x == 0; }); };

    // Degenerate ranges give the first colour everywhere
    CHECK(all_first(png_colour_rows(arma::mat(1, 1, arma::fill::value(5.0)))));
    CHECK(all_first(png_colour_rows(arma::mat(4, 3, arma::fill::zeros))));
    CHECK(all_first(png_colour_rows(arma::mat(4, 3, arma::fill::value(nan)))));
    CHECK(all_first(png_colour_rows(arma::mat(4, 3, arma::fill::value(inf)))));
    CHECK(all_first(png_colour_rows(arma::mat(4, 3, arma::fill::zeros), nan, nan, true))); // log, no positive value
    CHECK(all_first(png_colour_rows(arma::mat(4, 3, arma::fill::randu), 0.5, 0.5)));        // user min == max
    CHECK(all_first(png_colour_rows(arma::mat(4, 3, arma::fill::zeros), 2.0, nan)));        // user min above the data

    // The auto range ignores NaN and +-Inf; NaN gets the first colour, +-Inf are clipped
    arma::mat d = {{0.0, 1.0, inf}, {-inf, nan, 0.5}};
    std::vector<int> r = png_colour_rows(d);
    CHECK(r == std::vector<int>{0, 0, last, 0, last, (last + 1) / 2});

    // Log scale: non-positive values and NaN get the first colour, 10 and 1000 span the map
    arma::mat dl = {{10.0, -1.0, 1000.0, nan, 0.0}};
    r = png_colour_rows(dl, nan, nan, true);
    CHECK(r == std::vector<int>{0, 0, last, 0, 0});

    // User range: values outside are clipped
    arma::mat du = {{-5.0, 0.0, 10.0, 20.0}};
    r = png_colour_rows(du, 0.0, 10.0);
    CHECK(r == std::vector<int>{0, 0, last, last});

    // Errors: inverted user range, empty matrix
    std::remove("test_png_err.png");
    CHECK_THROWS_AS(quadriga_lib::write_png<double>(arma::mat(4, 4, arma::fill::randu), "test_png_err.png", "jet", 2.0, 1.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::write_png<double>(arma::mat(), "test_png_err.png"), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::write_png<double>(arma::mat(0, 5), "test_png_err.png"), std::invalid_argument);
    CHECK_FALSE(std::filesystem::exists("test_png_err.png"));

    // Float input takes the same path
    arma::fmat df = {{1.0f, 1.0f}};
    quadriga_lib::write_png<float>(df, "test_png_f.png");
    std::vector<unsigned char> img;
    unsigned w = 0, h = 0;
    REQUIRE(lodepng::decode(img, w, h, "test_png_f.png") == 0);
    auto cmap = quadriga_lib::colormap("jet", true);
    CHECK((img[0] == cmap(0, 0) && img[1] == cmap(0, 1) && img[2] == cmap(0, 2)));
    std::remove("test_png_f.png");
}
