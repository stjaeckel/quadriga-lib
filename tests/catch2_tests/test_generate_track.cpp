// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// generate_track (also appending a node), set_orientation and the seed hash of the layout streams.
//
// Oracles: public test vectors of the 64-bit FNV-1a hash; closed forms for the geometry of linear and circular tracks
// (end point, centre, radius, sense of rotation, closure); the heading of a track from the steps between its positions
// (brute force from the positions, not from the orientation under test); for the street type the pattern of the turns
// with fixed street length (the corners of one turn are one chamfer apart, the streets are the street length apart) and
// bounds that follow from the number of turns for the statistics. The random streams are checked with fixed seeds: the
// first uniform draw of the generator with the documented seed gives the direction. No bound depends on the realisation
// of a seed, only on the sample size.

#include <catch2/catch_test_macros.hpp>

#include "qd_helpers_layout.hpp"
#include "qd_test_layout.hpp"
#include "quadriga_layout.hpp"

#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
const double D = arma::datum::pi / 180.0;
const double NaN = arma::datum::nan;
const double PI = arma::datum::pi;
using opts = std::unordered_map<std::string, std::vector<double>>;

bool has(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }

// 64-bit FNV-1a of a byte string
unsigned long long fnv1a(const std::string &s, unsigned long long h = 14695981039346656037ULL)
{
    for (unsigned char c : s)
        h = (h ^ c) * 1099511628211ULL;
    return h;
}

// First uniform draw of the generator that the stream "track|<name>" of a layout seed uses, as direction in [0, 2 pi)
double oracle_direction(long long seed, const std::string &name)
{
    std::string key;
    for (int i = 0; i < 8; ++i)
        key += (char)(((unsigned long long)seed >> (8 * i)) & 0xFFULL);
    key += "track|" + name;
    std::mt19937_64 rng(fnv1a(key) & 0x7FFFFFFFFFFFFFFFULL);
    return 2.0 * PI * (double)(rng() >> 11) / 9007199254740992.0;
}

double az_of(const quadriga_lib::layout_nodes &N, arma::uword r) { return N.direction.is_empty() ? NaN : N.direction(0, r); }
double speed_of(const quadriga_lib::layout_nodes &N, arma::uword r) { return N.speed.is_empty() ? 0.0 : N.speed[r]; }

// Headings of the steps of the rows first ... first + n - 1, from the positions
std::vector<double> step_headings(const quadriga_lib::layout_nodes &N, arma::uword first, arma::uword n)
{
    std::vector<double> h;
    for (arma::uword r = first; r + 1 < first + n; ++r)
        h.push_back(std::atan2(N.pos(1, r + 1) - N.pos(1, r), N.pos(0, r + 1) - N.pos(0, r)));
    return h;
}

double step_length(const quadriga_lib::layout_nodes &N, arma::uword r) { return arma::norm(N.pos.col(r + 1) - N.pos.col(r)); }

// The turns of a street: runs of steps whose heading differs from that of the step before; the index of the first step
// of each run, and the heading before and after it
struct corner
{
    arma::uword at;
    double before, after;
};

std::vector<corner> corners_of(const std::vector<double> &h)
{
    std::vector<corner> m;
    arma::uword last = 0;
    for (arma::uword k = 1; k < h.size(); ++k)
    {
        if (std::abs(qd_layout_wrap(h[k] - h[k - 1])) <= 1e-6)
            continue;
        if (!m.empty() && k == last + 1) // The steps of one corner are consecutive
            m.back().after = h[k];
        else
            m.push_back({k, h[k - 1], h[k]});
        last = k;
    }
    return m;
}

// A street track from the origin-like start point (1, 2, 3)
quadriga_lib::layout_nodes street(double length, const opts &o, double dir = 0.0, arma::sword seed = 11, double spacing = NaN)
{
    auto N = qd_layout_nodes_from({qd_layout_pt(1.0, 2.0, 3.0)});
    quadriga_lib::generate_track(N, {0}, "street", length, 10.0, arma::vec{dir}, spacing, o, seed);
    return N;
}

// The two nodes of the fixture: UT1 (three rows, first position (100, 200, 1.5)) and UT2 (a point drop)
quadriga_lib::layout_nodes rx() { return qd_layout_build().rx; }
} // namespace

TEST_CASE("layout seed hash")
{
    // The public test vectors of the hash
    CHECK(fnv1a("") == 0xcbf29ce484222325ULL);
    CHECK(fnv1a("a") == 0xaf63dc4c8601ec8cULL);
    CHECK(fnv1a("foobar") == 0x85944171f73967e8ULL);

    for (long long base : {0LL, 1LL, 42LL, 123456789012345LL, -1LL})
        for (const std::string key : {"track|UT1", "", "track|", "x"})
        {
            std::string bytes;
            for (int i = 0; i < 8; ++i)
                bytes += (char)(((unsigned long long)base >> (8 * i)) & 0xFFULL);
            const long long e = (long long)(fnv1a(bytes + key) & 0x7FFFFFFFFFFFFFFFULL);
            CHECK(qd_layout_seed_hash(base, key) == e);
            CHECK(qd_layout_seed_hash(base, key) >= 0);
        }
    CHECK(qd_layout_seed_hash(5, "track|UT1") != qd_layout_seed_hash(5, "track|UT2"));
    CHECK(qd_layout_seed_hash(5, "track|UT1") != qd_layout_seed_hash(6, "track|UT1"));
    CHECK(qd_layout_seed_hash(5, "track|UT1") == qd_layout_seed_hash(5, "track|UT1"));
}

TEST_CASE("generate_track - input validation")
{
    auto N = rx();
    N.speed.subvec(0, 2).fill(2.0);
    const auto N0 = N;
    auto fails = [&](const std::string &part, auto call)
    {
        const std::string msg = qd_layout_throws(call);
        INFO("message: " << msg << ", expected part: " << part);
        CHECK(has(msg, part));
        CHECK(qd_layout_diff_nodes(N, N0).empty()); // A failed call leaves the nodes unchanged
        CHECK(N.is_valid() == "");
    };
    using quadriga_lib::generate_track;

    SECTION("type and nodes")
    {
        fails("type", [&] { generate_track(N, {0}, "spiral", 10.0, 1.0); });
        fails("type", [&] { generate_track(N, {0}, "", 10.0, 1.0); });
        fails("i_node", [&] { generate_track(N, {2}, "linear", 10.0, 1.0); });
        fails("i_node", [&] { generate_track(N, {0, 7}, "linear", 10.0, 1.0); });
        auto M = N;
        M.orientation.zeros(3, 2);
        CHECK(has(qd_layout_throws([&] { generate_track(M, {0}, "linear", 10.0, 1.0); }), "nodes"));
    }
    SECTION("numbers")
    {
        fails("length", [&] { generate_track(N, {0}, "linear", -1.0, 1.0); });
        fails("length", [&] { generate_track(N, {0}, "circular", NaN, 1.0); });
        fails("length", [&] { generate_track(N, {0}, "street", arma::datum::inf, 1.0); });
        fails("speed", [&] { generate_track(N, {0}, "linear", 10.0, -1.0); });
        fails("speed", [&] { generate_track(N, {0}, "point", 0.0, arma::datum::inf); });
        fails("speed", [&] { generate_track(N, {0}, "linear", 10.0, 0.0); }); // A track that moves needs a speed
        fails("spacing", [&] { generate_track(N, {0}, "linear", 10.0, 1.0, {}, 0.0); });
        fails("spacing", [&] { generate_track(N, {0}, "linear", 10.0, 1.0, {}, -1.0); });
        fails("spacing", [&] { generate_track(N, {0}, "linear", 10.0, 1.0, {}, arma::datum::inf); });
        fails("spacing", [&] { generate_track(N, {0}, "linear", 1.0e9, 1.0, {}, 1.0e-3); }); // Too many points
        fails("spacing", [&] { generate_track(N, {0}, "street", 1.0e9, 1.0); });
        fails("direction", [&] { generate_track(N, {0}, "linear", 10.0, 1.0, arma::vec{0.1, 0.2}); });
        fails("direction", [&] { generate_track(N, {0, 1}, "linear", 10.0, 1.0, arma::vec{0.1, 0.2, 0.3}); });
        fails("direction", [&] { generate_track(N, {0}, "linear", 10.0, 1.0, arma::vec{NaN}); });
        fails("direction", [&] { generate_track(N, {0}, "point", 0.0, 1.0, arma::vec{arma::datum::inf}); });
        fails("seed", [&] { generate_track(N, {0}, "linear", 10.0, 1.0, {}, NaN, {}, -2); });
        fails("seed", [&] { generate_track(N, {0}, "point", 0.0, 1.0, {}, NaN, {}, -1000); });
    }
    SECTION("speed of a track")
    {
        auto M = N;
        M.speed.zeros();
        const auto M0 = M;
        const std::string msg = qd_layout_throws([&] { generate_track(M, {0}, "linear", 10.0); });
        CHECK(has(msg, "speed"));
        CHECK(has(msg, "'UT1'")); // The message names the node
        CHECK(qd_layout_diff_nodes(M, M0).empty());
        CHECK(qd_layout_throws([&] { generate_track(M, {0}, "point"); }).empty()); // A point drop may stay
    }
    SECTION("options")
    {
        fails("options", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"bogus", {1.0}}}); });
        fails("options", [&] { generate_track(N, {0}, "linear", 100.0, 1.0, {}, NaN, {{"turn_probability", {0.5}}}); });
        fails("options", [&] { generate_track(N, {0}, "point", 0.0, 1.0, {}, NaN, {{"curve_radius", {5.0}}}); });
        fails("turn_probability", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"turn_probability", {1.5}}}); });
        fails("turn_probability", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"turn_probability", {0.5, 0.5}}}); });
        fails("turn_probability", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"turn_probability", {}}}); });
        fails("curve_radius", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"curve_radius", {-1.0}}}); });
        fails("street_length_min", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"street_length_min", {NaN}}}); });
        for (const char *key : {"street_length_min", "street_length_mu", "street_length_std", "curve_radius", "turn_probability"})
        {
            fails(key, [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{key, {arma::datum::inf}}}); });
            fails(key, [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{key, {-arma::datum::inf}}}); });
        }
        // A minimum that no draw reaches is an invalid argument, not a run-time error
        CHECK_THROWS_AS(generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"street_length_min", {1.0e9}}}), std::invalid_argument);
        CHECK(N.is_valid() == "");
        CHECK(qd_layout_diff_nodes(N, N0).empty());
        fails("street_length_std", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"street_length_std", {-1.0}}}); });
        // No street can be drawn: constant length 40 m below the minimum 50 m, or constant length 0
        fails("street_length", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"street_length_std", {0.0}}, {"street_length_mu", {40.0}}}); });
        fails("street_length", [&] { generate_track(N, {0}, "street", 100.0, 1.0, {}, NaN, {{"street_length_std", {0.0}}, {"street_length_mu", {0.0}}, {"street_length_min", {0.0}}}); });
    }
    SECTION("directions of the nodes of one site")
    {
        auto T = qd_layout_build().tx; // BS1 has two sectors
        const auto T0 = T;
        CHECK(has(qd_layout_throws([&] { generate_track(T, {0, 1}, "point", 0.0, 1.0, arma::vec{0.1, 0.2}); }), "direction"));
        CHECK(qd_layout_diff_nodes(T, T0).empty());
        CHECK(qd_layout_throws([&] { generate_track(T, {0, 1}, "point", 0.0, 1.0, arma::vec{0.1, 0.1}); }).empty());
    }
}

TEST_CASE("generate_track - point drops")
{
    auto N = rx();
    const auto N0 = N;
    N.time = {0.0, 1.0, 2.0, NaN};

    SECTION("one row with speed and an explicit horizontal direction")
    {
        quadriga_lib::generate_track(N, {0}, "point", 123.0, 2.0, arma::vec{30.0 * D});
        REQUIRE(N.n_node() == 2);
        CHECK(N.n_row() == 2);
        CHECK(N.n_row(0) == 1);
        CHECK(arma::all(N.offset == arma::uvec{0, 1}));
        CHECK(arma::abs(N.pos.col(0) - arma::vec{100.0, 200.0, 1.5}).max() == 0.0); // The first position
        CHECK(speed_of(N, 0) == 2.0);
        CHECK(std::abs(az_of(N, 0) - 30.0 * D) < 1e-15);
        CHECK(N.direction(1, 0) == 0.0);
        CHECK(N.time.is_empty()); // No node has a time any more
        CHECK(arma::abs(N.orientation.col(0)).max() == 0.0);
        CHECK(N.is_valid().empty());
        // The other node is untouched
        CHECK(arma::abs(N.pos.col(1) - N0.pos.col(3)).max() == 0.0);
        CHECK(speed_of(N, 1) == 1.5);
        CHECK(std::abs(az_of(N, 1) - 120.0 * D) < 1e-15);
        CHECK(std::abs(N.orientation(2, 1) - 45.0 * D) < 1e-15);
    }
    SECTION("NaN speed keeps a valid speed; the direction is wrapped")
    {
        quadriga_lib::generate_track(N, {1}, "point", 0.0, NaN, arma::vec{200.0 * D});
        CHECK(speed_of(N, 3) == 1.5);
        CHECK(std::abs(az_of(N, 3) - (-160.0 * D)) < 1e-13);
        CHECK(N.n_row(0) == 3); // The track UT1 is not in i_node
        CHECK(N.time.n_elem == 4);
    }
    SECTION("a node without speed gets 0")
    {
        quadriga_lib::generate_track(N, {0}, "point", 0.0, NaN, arma::vec{0.0});
        CHECK(speed_of(N, 0) == 0.0);
        CHECK(std::abs(az_of(N, 0)) < 1e-15);
    }
    SECTION("random direction: the first uniform draw of the stream of the node")
    {
        quadriga_lib::generate_track(N, {0, 1}, "point", 0.0, 1.0, {}, NaN, {}, 5);
        CHECK(std::abs(qd_layout_wrap(az_of(N, 0) - oracle_direction(5, "UT1"))) < 1e-12);
        CHECK(std::abs(qd_layout_wrap(az_of(N, 1) - oracle_direction(5, "UT2"))) < 1e-12);
        CHECK(N.direction(1, 0) == 0.0);
        CHECK(N.direction(1, 1) == 0.0);
        // Same seed, same result; another seed, another direction
        auto M = N0;
        quadriga_lib::generate_track(M, {0, 1}, "point", 0.0, 1.0, {}, NaN, {}, 5);
        CHECK(M.direction(0, 0) == N.direction(0, 0));
        quadriga_lib::generate_track(M, {0, 1}, "point", 0.0, 1.0, {}, NaN, {}, 6);
        CHECK(M.direction(0, 0) != N.direction(0, 0));
        CHECK(std::abs(qd_layout_wrap(az_of(M, 0) - oracle_direction(6, "UT1"))) < 1e-12);
    }
    SECTION("seed -1 uses the system random device")
    {
        // The two directions are drawn independently; they are equal with a probability of 2^-53
        quadriga_lib::generate_track(N, {0, 1}, "point", 0.0, 1.0, {}, NaN, {}, -1);
        for (arma::uword r = 0; r < 2; ++r)
            CHECK(std::abs(az_of(N, r)) <= PI);
        CHECK(az_of(N, 0) != az_of(N, 1));
    }
    SECTION("a node that moves keeps its orientation, also when mounted")
    {
        N.relative = {0, 1};
        N.orientation.col(3) = arma::vec{0.1, 0.2, 0.3};
        quadriga_lib::generate_track(N, {1}, "point", 0.0, 4.0, arma::vec{PI / 2.0});
        CHECK(arma::abs(N.orientation.col(3) - arma::vec{0.1, 0.2, 0.3}).max() == 0.0);
        CHECK(N.is_valid().empty());
        arma::mat p, o;
        quadriga_lib::sample_track(N, 1, arma::vec{0.0, 2.0}, p, o);
        CHECK(std::abs(p(1, 1) - (70.0 + 8.0)) < 1e-12);
    }
    SECTION("the call applies to all nodes of the site, each keeps its first position")
    {
        auto T = qd_layout_build().tx; // BS1 sectors 0 and 1 at (10, 10, 25), BS2 at (-300, 50, 30)
        T.pos(2, 1) = 26.0;
        quadriga_lib::generate_track(T, {1}, "point", 0.0, 3.0, arma::vec{1.0});
        REQUIRE(T.n_row() == 3);
        CHECK(std::abs(az_of(T, 0) - 1.0) < 1e-15);
        CHECK(std::abs(az_of(T, 1) - 1.0) < 1e-15);
        CHECK(speed_of(T, 0) == 3.0);
        CHECK(speed_of(T, 1) == 3.0);
        CHECK(std::isnan(az_of(T, 2)));
        CHECK(speed_of(T, 2) == 0.0);
        CHECK(T.pos(2, 1) == 26.0);
        CHECK(T.pos(2, 0) == 25.0);
    }
    SECTION("a direction per index")
    {
        quadriga_lib::generate_track(N, {0, 1}, "point", 0.0, 1.0, arma::vec{0.25, -0.5});
        CHECK(std::abs(az_of(N, 0) - 0.25) < 1e-15);
        CHECK(std::abs(az_of(N, 1) + 0.5) < 1e-15);
    }
    SECTION("the other members of the rows follow the first row")
    {
        auto M = rx();
        M.scenario = {"A", "B", "C", "D"};
        M.indoor = {1, 1, 1, 0};
        M.extra_names = {"note"};
        M.extra = {{"p", "q", "r", "s"}};
        quadriga_lib::generate_track(M, {0}, "point", 0.0, 1.0, arma::vec{0.0});
        CHECK(M.scenario == std::vector<std::string>{"A", "D"});
        CHECK(arma::all(M.indoor == arma::s32_vec{1, 0}));
        CHECK(M.extra[0] == std::vector<std::string>{"p", "s"});
    }
}

TEST_CASE("generate_track - linear tracks")
{
    auto N = rx();
    const arma::vec p0 = {100.0, 200.0, 1.5};
    const double d = 30.0 * D;
    const arma::vec e = {std::cos(d), std::sin(d), 0.0};

    SECTION("two points, speed, heading and no direction")
    {
        N.scenario = {"A", "B", "C", "D"};
        N.time = {0.0, 1.0, 2.0, NaN};
        quadriga_lib::generate_track(N, {0}, "linear", 100.0, 3.0, arma::vec{d});
        REQUIRE(N.n_node() == 2);
        CHECK(N.n_row(0) == 2);
        CHECK(N.n_row(1) == 1);
        CHECK(N.n_row() == 3);
        CHECK(arma::all(N.offset == arma::uvec{0, 2}));
        CHECK(arma::abs(N.pos.col(0) - p0).max() == 0.0);
        CHECK(arma::abs(N.pos.col(1) - (p0 + 100.0 * e)).max() < 1e-12);
        CHECK(speed_of(N, 0) == 3.0);
        CHECK(speed_of(N, 1) == 3.0);
        CHECK(speed_of(N, 2) == 1.5); // The other node
        CHECK(std::isnan(az_of(N, 0)));
        CHECK(std::isnan(az_of(N, 1)));
        CHECK(N.time.is_empty());
        for (arma::uword r = 0; r < 2; ++r)
        {
            CHECK(std::abs(N.orientation(0, r)) == 0.0);
            CHECK(std::abs(N.orientation(1, r)) < 1e-12);
            CHECK(std::abs(N.orientation(2, r) - d) < 1e-12);
        }
        CHECK(N.scenario == std::vector<std::string>{"A", "A", "D"});
        CHECK(N.is_valid().empty());
        // The other node is untouched
        CHECK(arma::abs(N.pos.col(2) - arma::vec{-50.0, 70.0, 1.5}).max() == 0.0);
        CHECK(std::abs(az_of(N, 2) - 120.0 * D) < 1e-15);
    }
    SECTION("the length divided by the spacing gives the number of points")
    {
        quadriga_lib::generate_track(N, {0}, "linear", 100.0, 3.0, arma::vec{d}, 10.0);
        REQUIRE(N.n_row(0) == 11);
        for (arma::uword k = 0; k < 11; ++k)
            CHECK(arma::abs(N.pos.col(k) - (p0 + 10.0 * (double)k * e)).max() < 1e-12);
        // 100 m with a spacing of at most 30 m: 4 steps of 25 m
        quadriga_lib::generate_track(N, {0}, "linear", 100.0, 3.0, arma::vec{d}, 30.0);
        REQUIRE(N.n_row(0) == 5);
        for (arma::uword k = 0; k < 5; ++k)
            CHECK(arma::abs(N.pos.col(k) - (p0 + 25.0 * (double)k * e)).max() < 1e-12);
        CHECK(N.is_valid().empty());
        CHECK(N.n_row(1) == 1);
    }
    SECTION("default length 1 m, the speed of the first row if valid")
    {
        N.speed[0] = 2.0;
        quadriga_lib::generate_track(N, {0}, "linear", 0.0, NaN, arma::vec{0.0});
        REQUIRE(N.n_row(0) == 2);
        CHECK(std::abs(N.pos(0, 1) - 101.0) < 1e-13);
        CHECK(speed_of(N, 0) == 2.0);
        CHECK(speed_of(N, 1) == 2.0);
    }
    SECTION("random direction, the heading follows")
    {
        quadriga_lib::generate_track(N, {0}, "linear", 50.0, 1.0, {}, NaN, {}, 9);
        const double h = oracle_direction(9, "UT1");
        CHECK(std::abs(qd_layout_wrap(N.orientation(2, 0) - h)) < 1e-12);
        CHECK(arma::abs(N.pos.col(1) - (p0 + 50.0 * arma::vec{std::cos(h), std::sin(h), 0.0})).max() < 1e-12);
    }
    SECTION("the height is kept, also for a node at another height")
    {
        N.pos(2, 0) = 12.0;
        quadriga_lib::generate_track(N, {0}, "linear", 50.0, 1.0, arma::vec{0.5});
        CHECK(N.pos(2, 0) == 12.0);
        CHECK(N.pos(2, 1) == 12.0);
    }
    SECTION("the type is not case sensitive")
    {
        quadriga_lib::generate_track(N, {0}, "Linear", 5.0, 1.0, arma::vec{0.0});
        CHECK(N.n_row(0) == 2);
    }
}

TEST_CASE("generate_track - circular tracks")
{
    auto N = qd_layout_nodes_from({qd_layout_pt(1.0, 2.0, 3.0)});
    const arma::vec p0 = {1.0, 2.0, 3.0};

    auto check_circle = [&](const quadriga_lib::layout_nodes &M, double length, double start, bool ccw, arma::uword n)
    {
        const double r = length / (2.0 * PI);
        const arma::vec c = p0 - r * arma::vec{std::cos(start), std::sin(start), 0.0}; // Centre: the start point is at the angle 'start'
        REQUIRE(M.n_row(0) == n);
        CHECK(arma::all(M.pos.row(0) == M.pos.row(0))); // No NaN
        for (arma::uword k = 0; k < n; ++k)
        {
            CHECK(std::abs(arma::norm(M.pos.col(k).head(2) - c.head(2)) - r) < 1e-12 * std::max(1.0, r));
            CHECK(M.pos(2, k) == 3.0);
        }
        for (arma::uword k = 0; k + 1 < n; ++k) // Sense of rotation from the cross product of radius and step
        {
            const arma::vec a = M.pos.col(k) - c, s = M.pos.col(k + 1) - M.pos.col(k);
            const double cr = a[0] * s[1] - a[1] * s[0];
            CHECK((cr > 0.0) == ccw);
        }
        CHECK(arma::all(M.pos.col(n - 1) == M.pos.col(0))); // Closed, exactly
        for (arma::uword r2 = 0; r2 < 3; ++r2)
            CHECK(M.orientation(r2, n - 1) == M.orientation(r2, 0));
    };

    SECTION("default: 129 points, circumference 20 pi, start in the east of the circle")
    {
        auto M = N;
        quadriga_lib::generate_track(M, {0}, "circular", 0.0, 2.0, arma::vec{0.0});
        check_circle(M, 20.0 * PI, 0.0, true, 129);
        CHECK(std::abs(M.pos(0, 1) - 1.0) < 0.1); // Starts to the north: x stays, y increases
        CHECK(M.pos(1, 1) > 2.0);
        CHECK(M.is_valid().empty());
        // The polygon of 128 sides is a little shorter than the circle
        double len = 0.0;
        for (arma::uword k = 0; k < 128; ++k)
            len += step_length(M, k);
        CHECK(len < 20.0 * PI);
        CHECK(len > 20.0 * PI * (1.0 - 2.0e-4));
        // Orientation: the heading of the steps, the last row repeats the first
        const auto h = step_headings(M, 0, 129);
        for (arma::uword k = 0; k < 128; ++k)
        {
            CHECK(std::abs(qd_layout_wrap(M.orientation(2, k) - h[k])) < 1e-12);
            CHECK(std::abs(M.orientation(1, k)) < 1e-12);
            CHECK(M.orientation(0, k) == 0.0);
        }
        CHECK(std::abs(M.orientation(2, 0) - (PI / 2.0 + PI / 128.0)) < 1e-12);
    }
    SECTION("a positive direction runs counter-clockwise from that angle, a negative one clockwise")
    {
        auto M = N;
        quadriga_lib::generate_track(M, {0}, "circular", 100.0, 2.0, arma::vec{0.7});
        check_circle(M, 100.0, 0.7, true, 129);
        M = N;
        quadriga_lib::generate_track(M, {0}, "circular", 100.0, 2.0, arma::vec{-0.7});
        check_circle(M, 100.0, -0.7, false, 129);
        CHECK(std::abs(qd_layout_wrap(M.orientation(2, 0) - (-0.7 - PI / 2.0 - PI / 128.0))) < 1e-12);
    }
    SECTION("a direction above pi runs counter-clockwise, one below -pi clockwise")
    {
        auto M = N;
        quadriga_lib::generate_track(M, {0}, "circular", 100.0, 2.0, arma::vec{3.5});
        check_circle(M, 100.0, 3.5, true, 129);
        M = N;
        quadriga_lib::generate_track(M, {0}, "circular", 100.0, 2.0, arma::vec{-3.5});
        check_circle(M, 100.0, -3.5, false, 129);
    }
    SECTION("a random direction is drawn in [0, 2 pi): the circle runs counter-clockwise")
    {
        for (int i = 0; i < 8; ++i)
        {
            auto M = N;
            M.name[0] = "C" + std::to_string(i);
            quadriga_lib::generate_track(M, {0}, "circular", 100.0, 2.0, {}, NaN, {}, 3);
            check_circle(M, 100.0, oracle_direction(3, M.name[0]), true, 129);
        }
    }
    SECTION("spacing")
    {
        auto M = N;
        quadriga_lib::generate_track(M, {0}, "circular", 62.0, 2.0, arma::vec{1.0}, 5.0); // ceil(62 / 5) = 13 steps
        check_circle(M, 62.0, 1.0, true, 14);
        for (arma::uword k = 0; k < 13; ++k)
            CHECK(step_length(M, k) <= 5.0);
        // Not fewer than 3 steps
        M = N;
        quadriga_lib::generate_track(M, {0}, "circular", 10.0, 2.0, arma::vec{1.0}, 100.0);
        CHECK(M.n_row(0) == 4);
    }
}

TEST_CASE("generate_track - street tracks, geometry and options")
{
    SECTION("without turns the street is straight and has the length")
    {
        const double d = 40.0 * D;
        const auto N = street(500.0, {{"turn_probability", {0.0}}}, d);
        REQUIRE(N.n_row(0) == 501);
        CHECK(arma::abs(N.pos.col(0) - arma::vec{1.0, 2.0, 3.0}).max() == 0.0);
        CHECK(arma::abs(N.pos.col(500) - (arma::vec{1.0, 2.0, 3.0} + 500.0 * arma::vec{std::cos(d), std::sin(d), 0.0})).max() < 1e-10);
        for (arma::uword k = 0; k < 500; ++k)
            CHECK(std::abs(step_length(N, k) - 1.0) < 1e-12);
        for (double h : step_headings(N, 0, 501))
            CHECK(std::abs(h - d) < 1e-12);
        CHECK(N.pos.row(2).min() == 3.0);
        CHECK(N.pos.row(2).max() == 3.0);
        for (arma::uword k = 0; k < 501; ++k)
        {
            CHECK(speed_of(N, k) == 10.0);
            CHECK(std::abs(N.orientation(2, k) - d) < 1e-12);
        }
        CHECK(N.is_valid().empty());
        // Spacing
        const auto S = street(500.0, {{"turn_probability", {0.0}}}, d, 11, 2.5);
        REQUIRE(S.n_row(0) == 201);
        CHECK(std::abs(step_length(S, 0) - 2.5) < 1e-12);
        const auto T = street(500.0, {{"turn_probability", {0.0}}}, d, 11, 3.0); // ceil(500 / 3) = 167 steps of at most 3 m
        REQUIRE(T.n_row(0) == 168);
        CHECK(step_length(T, 0) <= 3.0);
        CHECK(arma::norm(T.pos.col(167) - N.pos.col(500)) < 1e-10);
    }
    SECTION("the default length is 1000 m")
    {
        auto N = qd_layout_nodes_from({qd_layout_pt(0.0, 0.0, 0.0)});
        quadriga_lib::generate_track(N, {0}, "street", 0.0, 1.0, arma::vec{0.0}, NaN, {{"turn_probability", {0.0}}}, 3);
        CHECK(N.n_row(0) == 1001);
    }
    SECTION("fixed street length of 100 m, every crossing a turn: streets 100 m apart, chamfers of 2 r cos(turn / 2)")
    {
        // Streets of 100 m (std 0), a turn at every crossing, curve radius 10 m: the track runs straight for 100 m, cuts the
        // corner along r (u_old + u_new), whose length is 2 r cos(turn / 2) for a turn of 75 ... 105 deg, i.e. 12 ... 17.4 m
        // (turns of 45 ... 135 deg: 7.6 ... 18.4 m), and runs straight again. Every turn has two corners.
        const double L = 2000.0;
        const auto N = street(L, {{"street_length_std", {0.0}}, {"street_length_mu", {100.0}}, {"turn_probability", {1.0}}});
        const auto h = step_headings(N, 0, N.n_row(0));
        const auto c = corners_of(h);
        // Street 17 ends at 1700 + 16 chamfers (125 ... 294 m); the 18th street ends beyond 2000 m with 1800 + 17 * 7.6 = 1929 m at the least
        REQUIRE(c.size() >= 32);
        REQUIRE(c.size() <= 36);
        double sum_turn = 0.0;
        arma::uword n_turn = 0;
        for (arma::uword i = 0; i + 1 < c.size(); i += 2)
        {
            const double chamfer = (double)c[i + 1].at - (double)c[i].at;
            CHECK(chamfer >= 5.0);
            CHECK(chamfer <= 20.0);
            sum_turn += std::abs(qd_layout_wrap(c[i + 1].after - c[i].before));
            ++n_turn;
            if (i + 2 < c.size())
            {
                const double street_gap = (double)c[i + 2].at - (double)c[i + 1].at;
                CHECK(street_gap >= 98.0);
                CHECK(street_gap <= 102.0);
            }
        }
        CHECK(c[0].at >= 98);
        CHECK(c[0].at <= 101); // The first crossing after 100 m
        // The turn is 90 deg with a standard deviation of 15 deg; the mean of 17 turns has 3.6 deg
        CHECK(n_turn >= 16);
        CHECK(sum_turn / (double)n_turn > 75.0 * D);
        CHECK(sum_turn / (double)n_turn < 105.0 * D);
        // The track keeps its length
        double len = 0.0;
        for (arma::uword k = 0; k + 1 < N.n_row(0); ++k)
            len += step_length(N, k);
        CHECK(len <= L + 1e-9);
        CHECK(len >= 0.99 * L);
        CHECK(N.n_row(0) == 2001);
        // The road never turns back: the heading stays within 105 deg of the start direction
        for (double x : h)
            CHECK(std::abs(x) < 105.0 * D + 1e-9);
    }
    SECTION("curve radius")
    {
        for (double radius : {5.0, 20.0})
        {
            const auto N = street(2000.0, {{"street_length_std", {0.0}}, {"street_length_mu", {100.0}}, {"turn_probability", {1.0}}, {"curve_radius", {radius}}}, 0.0, 12);
            const auto c = corners_of(step_headings(N, 0, N.n_row(0)));
            REQUIRE(c.size() >= 4);
            for (arma::uword i = 0; i + 1 < c.size(); i += 2)
            {
                const double chamfer = (double)c[i + 1].at - (double)c[i].at;
                CHECK(chamfer >= 0.4 * radius - 2.0);
                CHECK(chamfer <= 2.0 * radius + 2.0);
            }
        }
    }
    SECTION("minimum street length")
    {
        // Median 60 m below the minimum 80 m: the lengths below 80 m are drawn again, the mean of the rest is 90 m
        const auto N = street(3000.0, {{"street_length_min", {80.0}}, {"street_length_mu", {60.0}}, {"street_length_std", {20.0}}, {"turn_probability", {1.0}}}, 0.0, 4);
        const auto c = corners_of(step_headings(N, 0, N.n_row(0)));
        REQUIRE(c.size() >= 20);
        double sum = 0.0;
        arma::uword n = 0;
        for (arma::uword i = 1; i + 1 < c.size(); i += 2)
        {
            const double gap = (double)c[i + 1].at - (double)c[i].at;
            CHECK(gap >= 78.0);
            sum += gap;
            ++n;
        }
        CHECK(sum / (double)n > 80.0);
        CHECK(sum / (double)n < 110.0);
    }
    SECTION("the direction rotates the whole track")
    {
        const auto A = street(300.0, {}, 0.0, 21);
        const auto B = street(300.0, {}, 0.7, 21);
        REQUIRE(A.n_row(0) == B.n_row(0));
        // The same random street, rotated by 0.7 rad around the start point
        for (arma::uword k = 0; k < A.n_row(0); ++k)
        {
            const double x = A.pos(0, k) - 1.0, y = A.pos(1, k) - 2.0;
            CHECK(std::abs(B.pos(0, k) - 1.0 - (x * std::cos(0.7) - y * std::sin(0.7))) < 1e-9);
            CHECK(std::abs(B.pos(1, k) - 2.0 - (x * std::sin(0.7) + y * std::cos(0.7))) < 1e-9);
        }
    }
}

TEST_CASE("generate_track - street tracks, statistics and seeds")
{
    SECTION("turn probability: the number of crossings with a turn")
    {
        // Streets of exactly 50 m and a curve radius of 0: a turn is one corner at the end of a street, and 50 000 m hold
        // 999 crossings (at 50 k m, k = 1 ... 999). The number of turns is binomial(999, p): for p = 0.5 mean 499.5 and
        // standard deviation 15.8; the bound is 4 sigma = 63, which p = 0.4 (400) and p = 0.6 (600) would violate
        const auto N = street(50000.0, {{"street_length_std", {0.0}}, {"street_length_mu", {50.0}}, {"turn_probability", {0.5}}, {"curve_radius", {0.0}}}, 0.0, 31);
        const auto h = step_headings(N, 0, N.n_row(0));
        const auto c = corners_of(h);
        CHECK((double)c.size() > 499.5 - 63.0);
        CHECK((double)c.size() < 499.5 + 63.0);
        for (double x : h)
            CHECK(std::abs(x) < 105.0 * D + 1e-9); // The road never turns back
    }
    SECTION("the first turn of many streets: size and side")
    {
        // 800 streets (own names, own streams) with a turn at 50 m, the start direction is 0. The new direction is 90 deg
        // +- 15 deg * z for a left (right) turn, z standard normal, and must stay within 105 deg of the start: z < 1, a normal
        // distribution truncated at +1 sigma. With r = phi(1) / Phi(1) it has the mean 90 - 15 r = 85.7 deg and the standard
        // deviation 15 sqrt(1 - r - r^2) = 11.9 deg. The bounds are 4 standard errors; the side is 0.5 +- 4 * 0.5 / sqrt(800)
        const int n = 800;
        std::vector<arma::mat> tracks(n, qd_layout_pt(0.0, 0.0, 0.0));
        auto M = qd_layout_nodes_from(tracks);
        quadriga_lib::generate_track(M, quadriga_lib::uint_list(arma::regspace<arma::uvec>(0, n - 1)), "street", 60.0, 1.0, arma::vec{0.0}, NaN,
                                     {{"street_length_std", {0.0}}, {"street_length_mu", {50.0}}, {"turn_probability", {1.0}}, {"curve_radius", {0.0}}}, 5);
        double sum = 0.0, sum2 = 0.0;
        int left = 0;
        for (arma::uword i = 0; i < (arma::uword)n; ++i)
        {
            const auto h = step_headings(M, M.offset[i], M.n_row(i));
            const double t = qd_layout_wrap(h[55] - h[40]);
            REQUIRE(std::abs(t) > 1e-3); // Every street has its turn
            left += t > 0.0 ? 1 : 0;
            sum += std::abs(t);
            sum2 += t * t;
        }
        const double r = std::exp(-0.5) / std::sqrt(2.0 * PI) / (0.5 * std::erfc(-1.0 / std::sqrt(2.0)));
        const double m = (90.0 - 15.0 * r) * D, sd = 15.0 * D * std::sqrt(1.0 - r - r * r);
        const double mean = sum / n, sdev = std::sqrt(sum2 / n - mean * mean);
        CHECK(std::abs(mean - m) < 4.0 * sd / std::sqrt((double)n));
        CHECK(std::abs(sdev - sd) < 4.0 * sd / std::sqrt(2.0 * (n - 1)));
        CHECK(std::abs((double)left / n - 0.5) < 4.0 * 0.5 / std::sqrt((double)n));
    }
    SECTION("a given direction does not change the shape of the street")
    {
        // The start direction consumes its draw also when it is given: the street of a seed is the same, rotated
        const auto A = street(300.0, {}, 0.0, 21);
        auto B = qd_layout_nodes_from({qd_layout_pt(1.0, 2.0, 3.0)});
        quadriga_lib::generate_track(B, {0}, "street", 300.0, 10.0, {}, NaN, {}, 21);
        const double h0 = oracle_direction(21, "N0");
        REQUIRE(A.n_row(0) == B.n_row(0));
        for (arma::uword k = 0; k < A.n_row(0); ++k)
        {
            const double x = A.pos(0, k) - 1.0, y = A.pos(1, k) - 2.0;
            CHECK(std::abs(B.pos(0, k) - 1.0 - (x * std::cos(h0) - y * std::sin(h0))) < 1e-9);
            CHECK(std::abs(B.pos(1, k) - 2.0 - (x * std::sin(h0) + y * std::cos(h0))) < 1e-9);
        }
    }
    SECTION("the same seed gives the same street, another seed another one")
    {
        const opts o = {{"turn_probability", {1.0}}};
        const auto A = street(800.0, o, 0.3, 7);
        const auto B = street(800.0, o, 0.3, 7);
        const auto C = street(800.0, o, 0.3, 8);
        CHECK(arma::abs(A.pos - B.pos).max() == 0.0);
        CHECK(arma::abs(A.pos - C.pos).max() > 1.0);
    }
    SECTION("the sectors of one site share the draws, other names get other draws")
    {
        auto T = qd_layout_build().tx;
        T.pos(2, 1) = 26.0;
        quadriga_lib::generate_track(T, {0}, "street", 300.0, 5.0, {}, NaN, {{"turn_probability", {1.0}}}, 15);
        REQUIRE(T.n_node() == 3);
        REQUIRE(T.n_row(0) == T.n_row(1));
        for (arma::uword k = 0; k < T.n_row(0); ++k)
        {
            const arma::vec a = T.pos.col(T.offset[0] + k) - T.pos.col(T.offset[0]);
            const arma::vec b = T.pos.col(T.offset[1] + k) - T.pos.col(T.offset[1]);
            CHECK(arma::abs(a - b).max() < 1e-12);
        }
        CHECK(T.pos(2, T.offset[1]) == 26.0);
        CHECK(T.n_row(2) == 1);
        // A site with another name gets its own start direction
        auto M = qd_layout_nodes_from({qd_layout_pt(0.0, 0.0, 0.0), qd_layout_pt(0.0, 0.0, 0.0)});
        quadriga_lib::generate_track(M, {0, 1}, "linear", 10.0, 1.0, {}, NaN, {}, 15);
        CHECK(std::abs(qd_layout_wrap(M.orientation(2, 0) - oracle_direction(15, "N0"))) < 1e-12);
        CHECK(std::abs(qd_layout_wrap(M.orientation(2, 2) - oracle_direction(15, "N1"))) < 1e-12);
        CHECK(std::abs(M.orientation(2, 0) - M.orientation(2, 2)) > 1e-6);
    }
    SECTION("seed -1: a valid street from the system random device")
    {
        const auto N = street(200.0, {}, 0.0, -1);
        CHECK(N.n_row(0) == 201);
        CHECK(N.is_valid().empty());
        CHECK(arma::all(N.pos.col(0) == arma::vec{1.0, 2.0, 3.0}));
    }
}

TEST_CASE("generate_track - an empty i_node appends a node")
{
    quadriga_lib::layout L = qd_layout_build();
    quadriga_lib::layout_nodes N = L.rx; // UT1 (3 rows), UT2 (1 row)
    const quadriga_lib::layout_nodes N0 = N;

    // A linear track for a new node at the origin, named after the existing names
    quadriga_lib::generate_track(N, {}, "linear", 10.0, 2.0, {0.0}, NaN, {}, 1);
    REQUIRE(N.n_node() == 3);
    const arma::uword k = N.n_node() - 1; // The index of the new node
    CHECK(N.name[k] == "UT3");
    CHECK(N.sector[k] == 0);
    CHECK(N.i_antenna[k] == 0);
    CHECK(N.offset[k] == 4);
    REQUIRE(N.n_row(k) == 2);
    CHECK(arma::approx_equal(N.pos.cols(4, 5), arma::mat{{0.0, 10.0}, {0.0, 0.0}, {0.0, 0.0}}, "absdiff", 1e-12));
    CHECK(arma::approx_equal(N.speed.subvec(4, 5), arma::vec{2.0, 2.0}, "absdiff", 0.0));
    CHECK(arma::approx_equal(N.pos.cols(0, 3), N0.pos, "absdiff", 0.0)); // The other nodes stay
    CHECK(qd_layout_same_angle(N.direction.cols(0, 3), N0.direction));
    CHECK(N.is_valid() == "");

    // A point drop with a name and a start position
    quadriga_lib::generate_track(N, {}, "point", 0.0, 1.5, {1.0}, NaN, {}, 1, "car", {5.0, 6.0, 1.5});
    REQUIRE(N.n_node() == 4);
    CHECK(N.name[3] == "car");
    CHECK(arma::approx_equal(N.pos.col(6), arma::vec{5.0, 6.0, 1.5}, "absdiff", 0.0));
    CHECK(N.speed[6] == 1.5);
    CHECK(std::abs(N.direction(0, 6) - 1.0) < 1e-12);
    CHECK(N.is_valid() == "");

    // Names: the next number of a common prefix, else N<n>
    quadriga_lib::layout_nodes T = L.tx; // BS1 (2 sectors), BS2
    quadriga_lib::generate_track(T, {}, "point", 0.0, 0.0, {0.0});
    CHECK(T.name.back() == "BS3");
    quadriga_lib::generate_track(N, {}, "point", 0.0, 0.0, {0.0}); // UT1, UT2, UT3, car
    CHECK(N.name.back() == "N1");
    quadriga_lib::layout_nodes E;
    quadriga_lib::generate_track(E, {}, "point", 0.0, 0.0, {0.0});
    quadriga_lib::generate_track(E, {}, "point", 0.0, 0.0, {0.0});
    CHECK(E.name == std::vector<std::string>{"N1", "N2"});
    CHECK(E.is_valid() == "");

    // Errors leave the nodes unchanged
    auto msg = [&](const quadriga_lib::uint_list &i, const std::string &type, double speed, const std::string &name, const arma::vec &start)
    {
        quadriga_lib::layout_nodes M = N0;
        const std::string m = qd_layout_throws([&] { quadriga_lib::generate_track(M, i, type, 0.0, speed, {0.0}, NaN, {}, 1, name, start); });
        CHECK(qd_layout_diff_nodes(M, N0) == "");
        return m;
    };
    CHECK(msg({0}, "point", 1.0, "car", {}) == "Inputs 'name' and 'start' are used only when 'i_node' is empty.");
    CHECK(msg({0}, "point", 1.0, "", {1.0, 2.0, 3.0}) == "Inputs 'name' and 'start' are used only when 'i_node' is empty.");
    CHECK(msg({}, "point", 1.0, "UT2", {}) == "Input 'name' is the name of an existing node ('UT2').");
    CHECK(msg({}, "point", 1.0, "", {1.0, 2.0}) == "Input 'start' must be empty or [x, y, z] with finite values.");
    CHECK(msg({}, "point", 1.0, "", {1.0, 2.0, NaN}) == "Input 'start' must be empty or [x, y, z] with finite values.");
    CHECK(has(msg({}, "point", 1.0, "a,b", {}), "Input 'name'"));
    CHECK(has(msg({}, "linear", NaN, "", {}), "speed")); // A track needs a speed: the new row has none
}

TEST_CASE("generate_track - mounted antennas")
{
    auto N = qd_layout_nodes_from({arma::mat{{0.0, 5.0, 9.0}, {0.0, 1.0, 7.0}, {0.0, 0.0, 0.0}}});
    N.orientation = arma::mat{{0.1, 0.1, 0.1}, {0.2, 0.2, 0.2}, {30.0 * D, 30.0 * D, 30.0 * D}};
    N.relative = {1};
    const arma::vec o = {0.1, 0.2, 30.0 * D};

    SECTION("a track keeps the orientation of its first row on all rows")
    {
        quadriga_lib::generate_track(N, {0}, "linear", 10.0, 1.0, arma::vec{90.0 * D});
        REQUIRE(N.n_row() == 2);
        CHECK(arma::abs(N.orientation.col(0) - o).max() == 0.0);
        CHECK(arma::abs(N.orientation.col(1) - o).max() == 0.0);
        CHECK(N.is_valid().empty());
        // At the snapshots the antenna follows the motion: heading 90 deg + 30 deg
        arma::mat p, ori;
        quadriga_lib::sample_track(N, 0, arma::vec{0.0, 5.0, 10.0}, p, ori);
        const arma::mat R = qd_layout_rot(0.0, 0.0, 90.0 * D) * qd_layout_rot(0.1, 0.2, 30.0 * D);
        for (arma::uword k = 0; k < 3; ++k)
            CHECK(arma::abs(qd_layout_rot(ori(0, k), ori(1, k), ori(2, k)) - R).max() < 1e-12);
        CHECK(std::abs(p(1, 2) - 10.0) < 1e-12);
    }
    SECTION("circular and street tracks too")
    {
        auto M = N;
        quadriga_lib::generate_track(M, {0}, "circular", 50.0, 1.0, arma::vec{0.0});
        for (arma::uword k = 0; k < M.n_row(); ++k)
            CHECK(arma::abs(M.orientation.col(k) - o).max() == 0.0);
        quadriga_lib::generate_track(N, {0}, "street", 50.0, 1.0, arma::vec{0.0}, NaN, {}, 2);
        for (arma::uword k = 0; k < N.n_row(); ++k)
            CHECK(arma::abs(N.orientation.col(k) - o).max() == 0.0);
    }
    SECTION("a point drop gets its direction, the orientation stays")
    {
        auto M = qd_layout_nodes_from({qd_layout_pt(0.0, 0.0, 0.0)});
        M.relative = {1};
        M.orientation.col(0) = o;
        CHECK_FALSE(M.is_valid().empty()); // Mounted, but no direction
        quadriga_lib::generate_track(M, {0}, "point", 0.0, 2.0, arma::vec{PI / 2.0});
        CHECK(M.is_valid().empty());
        CHECK(arma::abs(M.orientation.col(0) - o).max() == 0.0);
        arma::mat p, ori;
        quadriga_lib::sample_track(M, 0, arma::vec{1.0}, p, ori);
        CHECK(std::abs(p(1, 0) - 2.0) < 1e-12);
        CHECK(arma::abs(qd_layout_rot(ori(0, 0), ori(1, 0), ori(2, 0)) - qd_layout_rot(0.0, 0.0, PI / 2.0) * qd_layout_rot(0.1, 0.2, 30.0 * D)).max() < 1e-12);
    }
}
